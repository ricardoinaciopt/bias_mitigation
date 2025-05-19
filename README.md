# Simulating Bias for Mitigation Models and Fairness

## 1. Introduction and System Goals

This paper simulates a loan approval process using an Agent-Based Model (ABM) and subsequently evaluates the fairness and performance of machine learning models trained on the simulated data. The system facilitates the study of algorithmic bias under controlled conditions, allowing for the specification of **representation bias** (disparate proportions of different groups) and **label bias** (systemic advantages/disadvantages in qualification scoring based on group membership).

The core functionality revolves around generating synthetic applicant data reflecting a society with two groups ('A' - privileged, 'B' - protected) and then processing this data through two distinct machine learning pipelines:

1.  **Static Pipeline:** Simulates a traditional batch learning setup where a model (XGBoost) is trained once on the entire historical dataset.
2.  **Dynamic Pipeline:** Simulates an online learning setup where a model (River Hoeffding Tree) learns incrementally as new data arrives.

Both pipelines incorporate optional fairness interventions (reweighing, in-processing mitigation) and post-processing (calibration, explainability analysis).

The following components of this work can be found in the main file: `main_abm_pipeline.py`

## 2. Detailed ABM Simulation (`simulate_abm_ticks`)

This function orchestrates the core agent-based simulation.

1.  **Initialization (`random.seed`, `np.random.seed`):** Ensures simulation reproducibility given a `seed`.
2.  **State Variables:**
    * `agents`: Dictionary mapping agent `uid` to `Applicant` object for currently active agents.
    * `peer_net`: `networkx.Graph` representing social connections.
    * `processed_agents`: List to store `Applicant` objects after their loan decision is made.
    * `agent_uid_counter`: Integer to assign unique IDs.
3.  **Main Loop (`for _ in range(max_ticks):`)**: Executes for the specified number of simulation steps.
    * **Agent Arrival (`if random.random() < ARRIVAL_RATE:`):**
        * Determines group ('A' or 'B') based on `rep` parameter (`random.random() < rep`).
        * Instantiates `Applicant(agent_uid_counter, g, lbl)`. `lbl` (label bias) is passed here.
        * Agent joins active pool (`agents`, `peer_net`) only if `random.random() < a.trust` (stochastic entry based on initial trust).
        * If added, connects to up to 3 random existing agents in `peer_net`.
        * Increments `agent_uid_counter`.
    * **Peer Interaction Loop (`for uid, a in agents.items():`)**: Iterates through currently active agents.
        * **Trust Update:** Calculates average trust of neighbors (present in `agents`) and updates the agent's trust via an EMA-like formula (`a.trust = 0.9 * a.trust + 0.1 * np.mean(neighbor_trusts)`).
        * **Wealth Transfer:** With low probability (0.1), selects a random neighbor. If neighbor exists, transfers 5% of the minimum wealth between them.
    * **Processing/Waiting Loop (`for uid, a in list(agents.items()):`)**: Iterates through a copy of agent keys to allow deletion.
        * **Decision Trigger (`if not a.processed and a.waiting_time >= 3:`):** If agent is waiting and has waited >= 3 ticks:
            * Calculates qualification score: `a.qualification_score()` (incorporates `lbl_bias`).
            * Determines true qualification: `a.qualified = ... >= APPROVAL_THRESHOLD`.
            * Applies qualification noise: `if random.random() < NOISE_QUAL: a.qualified = not a.qualified`.
            * Sets loan approval status: `a.loan_approved = a.qualified`.
            * Applies consequences: `a.apply_decision_effect()`.
            * Adds agent to `processed_agents`.
        * **Increment Wait (`elif not a.processed:`):** If waiting but not time yet, increment `a.waiting_time`.
        * **Agent Removal (`elif a.processed and a.waiting_time > 15:`):** If processed and waited > 15 ticks post-processing, remove from active `agents` dictionary (`del agents[uid]`).
        * **Post-Processing Wait (`elif a.processed:`):** If processed but still within 15 ticks, increment `a.waiting_time`.
4.  **Return Value:** Returns the `processed_agents` list.

## 3. Detailed Pipeline Workflows

### 3.1. Static Pipeline (`run_static`)

Executes a batch learning process.

1.  **Log Start:** Records parameters used.
2.  **Data Generation:** Calls `simulate_abm_ticks(rep, lbl, seed, max_ticks=MAX_TICKS)`. Creates `df` DataFrame using `extract_features` on `processed_agents`. Handles empty `df`.
3.  **Feature/Target/Group Separation:** Extracts `X`, `y`, `groups` from `df`.
4.  **Data Splitting:** Uses `train_test_split` twice to create Train (60%), Calibration (20%), Test (20%) sets (`X_train`, `y_train`, `g_train`, etc.), attempting stratification by `y`. Includes fallback for non-stratifiable data.
5.  **Scaling:** Initializes `SklearnStandardScaler`. Fits *only* on `X_train` (`scaler.fit_transform`). Applies transformation to `X_cal` and `X_test` (`scaler.transform`). Converts results back to pandas DataFrames.
6.  **Reweighing Setup:** Initializes weights `w_train`, `w_cal` to `None`. If `reweight` is True:
    * **Manual:** If `group_weights` provided, maps group labels (`g_train`, `g_cal`) to weights. Stores `reweighting_mode="manual"`.
    * **Automatic:** If `group_weights` is `None`, uses `aif360.sklearn.preprocessing.Reweighing`. Instantiates `Reweighing(prot_attr=g_train/g_cal)`. Calls `fit_transform` on dataframes containing features and group columns to get `w_train`, `w_cal`. Stores `reweighting_mode="auto"`. Calculates average weights per group for metadata. Handles `ImportError` or other exceptions by disabling reweighing.
7.  **XGBoost Training:**
    * Defines `param_dist` for `RandomizedSearchCV`.
    * Initializes `xgb.XGBClassifier`.
    * Initializes `RandomizedSearchCV` wrapping the XGBoost model, using `param_dist`, `roc_auc` scoring, and cross-validation (`cv=3`).
    * Fits `random_search` on `X_train`, `y_train`, passing `sample_weight=w_train` if available.
    * Stores the `best_estimator_` found by the search (`base_estimator`).
8.  **Mitigation Application:**
    * Sets `final_estimator = base_estimator`.
    * If `use_mitigator` is True *and* `reweight` is False:
        * Instantiates `fairlearn.reductions.ExponentiatedGradient` using `base_estimator` and the specified `fairness_constraint` (`DemographicParity` or `EqualizedOdds`).
        * Fits the `mitigator` on `X_train`, `y_train`, `sensitive_features=g_train`.
        * Updates `final_estimator = mitigator`. Sets `mitigator_active = True`. Handles `ImportError` or other exceptions.
9.  **Calibration Application:**
    * Sets `calibrated_model = final_estimator`.
    * Determines if calibration is needed (`needs_calibration = not (mitigator_active and not reweight)`).
    * If `needs_calibration`:
        * Initializes `CalibratedClassifierCV(final_estimator, method="isotonic", cv="prefit")`.
        * Fits `calibrator` on `X_cal`, `y_cal`, passing `sample_weight=w_cal` if available.
        * Updates `calibrated_model = calibrator`. Sets `calibration_mode = "Isotonic"`. Handles exceptions.
10. **Test Set Evaluation:**
    * Selects the model to evaluate (`model_to_evaluate = calibrated_model`).
    * Predicts probabilities (`proba_test`) on `X_test` using `model_to_evaluate.predict_proba`. If `AttributeError` occurs (likely due to mitigator), attempts `ensemble_predict_proba`. Handles failures by setting `proba_test` to NaN.
    * Calls `_calculate_final_metrics(y_test, proba_test, g_test)` to get `overall_metrics`, `group_metrics`, `bias_metrics`.
11. **Explainability:**
    * Constructs `xai_tag` string based on parameters.
    * Calls `explainability_analysis(model=model_to_evaluate, X_train=X_train, X_test=X_test, output_dir=XAI_DIR, prefix=xai_tag)`. Handles potential errors.
12. **Saving Results:**
    * Constructs `meta` dictionary with detailed run configuration.
    * Constructs base `tag` string for filenames.
    * Saves metrics to JSON using `save_fairness_json` in `JSON_DIR`.
    * Saves the original generated `df` (before splitting/scaling) to CSV in `CSV_DIR`.
13. **Return:** Returns an empty DataFrame and the `overall_metrics` dictionary.

### 3.2. Dynamic Pipeline (`run_dynamic`)

Executes an online learning process with hyperparameter tuning.

1.  **Log Start:** Records parameters used.
2.  **Reproducibility:** Sets `numpy` and `random` seeds.
3.  **Hyperparameter Tuning:**
    * Defines `param_grid` for `OnlineApprovalClassifier` (Hoeffding Tree parameters).
    * Uses `ParameterSampler` to get `n_iter=10` parameter sets.
    * Initializes `best_score = -inf`, `best_params = None`.
    * **Loop:** For each `params` set:
        * Instantiates `tuning_model = OnlineApprovalClassifier(...)` with current `params`, `update_interval`, `calibrate_interval`, reweigh/mitigate flags.
        * Calls `simulate_dynamic_loop(tuning_model, rep, lbl, seed)`. This function internally:
            * Generates a full agent stream via `simulate_abm_ticks`.
            * Processes the stream via `_process_agents_online` using `tuning_model` (involves `river` learning, potential reweighing/mitigation/calibration updates).
            * Calculates final metrics for this run via `_calculate_final_metrics`.
        * Retrieves `overall_metrics_tune` from the result.
        * Gets `score = overall_metrics_tune.get("roc_auc", 0)` (handles NaN).
        * Updates `best_score` and `best_params` if `score` is higher.
    * Logs the best score and parameters found. Handles fallback if no best params identified.
4.  **Final Model Initialization:** Instantiates `final_model = OnlineApprovalClassifier(...)` using the determined `best_params`.
5.  **Final Data Generation:** Calls `simulate_abm_ticks(rep, lbl, seed, max_ticks=MAX_TICKS)` *once* to get the definitive `final_agents` stream.
6.  **Prepare Data for Saving/XAI:** Extracts features and metadata from `final_agents` into `df_final_agents_full`. Creates `X_full_df_for_xai` (features only) for explainability background.
7.  **Save Raw Data:** Saves `df_final_agents_full` to CSV in `CSV_DIR`. Handles empty data case.
8.  **Prepare Tags:** Constructs `xai_tag` and `tag_base` for outputs.
9.  **Final Online Processing:**
    * Calls `final_processed_results = _process_agents_online(...)` passing the `final_model`, `final_agents`, `calibrate_interval`, setting `run_explainability=True`, and providing `xai_tag` and `X_full_df_for_xai`. This function handles the sequential processing, model updates, periodic calibration/mitigation, metric tracking, and triggers the `explainability_analysis` call.
10. **Final Metric Calculation:** Calls `_calculate_final_metrics` using the `y_all`, `probas_all`, `groups_all` returned in `final_processed_results` to get `overall_metrics_final`, `group_metrics_final`, `bias_metrics_final`.
11. **Saving Results:**
    * Constructs `meta` dictionary including `best_online_params`.
    * Saves metrics (overall, group, bias, per-tick) and metadata to JSON using `save_fairness_json` in `JSON_DIR`.
12. **Return:** Returns a DataFrame of `metrics_per_tick` and the `overall_metrics_final` dictionary.

## 4. Summary and Conclusion

This framework provides a robust platform for investigating algorithmic fairness in a simulated loan approval process. By combining a configurable ABM (`simulate_abm_ticks`, `Applicant.py`) with distinct static (`run_static`) and dynamic (`run_dynamic`, `OnlineClassifier.py`) machine learning pipelines, it allows for comparative analysis of different learning paradigms under varying bias conditions.
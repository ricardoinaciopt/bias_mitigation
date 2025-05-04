## 1. Underlying “true” concept

* Both pipelines take the ABM’s **`a.qualified`** flag (after noise and a threshold on `qualification_score`) as the ground-truth “approved” label.
* In the ABM tick loop (`simulate_abm_ticks`), after 3 ticks each agent’s `a.qualified` is set and then copied into `a.loan_approved`—so **qualified** and **loan\_approved** are identical by construction.
* **Static** uses `int(a.qualified)` as its `y`-vector; **dynamic** uses `y_true = int(a.qualified)` in its online loop.

→ *No mismatch here: both are predicting the same “loan approval” outcome.*

---

## 2. What is being predicted: labels vs. probabilities

| Pipeline    | Prediction API                          | Returned value              | Label derivation                  |
| ----------- | --------------------------------------- | --------------------------- | --------------------------------- |
| **Static**  | `calibrator.predict_proba(X_test)[:,1]` | float ∈ \[0,1] (P(approve)) | `pred = (proba>=0.5).astype(int)` |
| **Dynamic** | `model.predict_proba(x_df)[0,1]`        | float ∈ \[0,1] (P(approve)) | `pred = int(proba >= 0.5)`        |

* **Both** produce a real-valued probability for class 1 (“approve”), and then threshold at **0.5** to get a binary prediction.
* They both collect the raw probabilities (`probas_all`) and true labels (`y_all`) so that metrics like ROC AUC, log-loss, accuracy, etc., can be computed.

---

## 3. Key differences in *how* those probabilities are produced

| Aspect                    | Static pipeline                                               | Dynamic (online) pipeline                                                                                       |
| ------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Model family**          | Offline XGBoost (`XGBClassifier`)                             | River’s incremental HoeffdingTree (`tree.HoeffdingTreeClassifier`)                                              |
| **Hyperparameter tuning** | `RandomizedSearchCV` over XGB params (10 iterations)          | `ParameterSampler` + grid of River tree params                                                                  |
| **Calibration**           | `CalibratedClassifierCV(method="isotonic", cv="prefit")`      | River’s `IsotonicRegression` fitted on rolling batches inside `_fit_calibration()`                              |
| **Fairness mitigation**   | Optional `ExponentiatedGradient` wrap around XGBoost post-fit | Optional rolling‐window `ExponentiatedGradient` inside `OnlineApprovalClassifier._fit_mitigator_rolling()`      |
| **Feature scaling**       | Batch `StandardScaler` from scikit-learn                      | Streaming `river_preprocessing.StandardScaler().learn_one()/transform_one()`                                    |
| **Model updates**         | None (purely train-then-test)                                 | Buffers up `update_interval` samples, then `_flush_buffers()` → tree learning + calibration + (maybe mitigator) |
| **Metrics collection**    | One static evaluation on held-out test split                  | Per-tick metrics in `_process_agents_online`, plus final aggregated metrics                                     |

---

## 4. Verdict

* **Are they predicting the same underlying concept?**
  Yes: both use the ABM’s `a.qualified` (≡ `loan_approved`) as ground truth.

* **Do they compute probabilities or labels (or both)?**
  Both pipelines produce real‐valued probabilities via `predict_proba(…)[…,1]` and then threshold at 0.5 to get binary predictions.

* **Any inconsistencies?**
  No—in terms of *what* is being predicted and *how* labels are derived, the two branches align perfectly. The only differences are in modeling strategy (offline vs. online), calibration, and fairness‐mitigation workflows.


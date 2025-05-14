from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterSampler,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from aif360.sklearn.preprocessing import Reweighing

from Applicant import Applicant
from OnlineClassifier import OnlineApprovalClassifier
from explainability_wrapper import explainability_analysis


MAX_TICKS = 10000
ARRIVAL_RATE = 0.8
APPROVAL_THRESHOLD = 30
NOISE_QUAL = 0.05


def get_dirs(setting: str):
    """Return CSV, JSON, XAI directory names based on setting."""
    suffix = "_dynamic" if setting == "dynamic" else "_static"
    return (
        f"abm_csv{suffix}",
        f"abm_json{suffix}",
        f"explainability_results{suffix}",
    )


def replace_nans(obj: Any) -> Any:
    """Recursively replace NaN values with None in dicts/lists for JSON compatibility."""
    if isinstance(obj, dict):
        return {k: replace_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nans(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def ensemble_predict_proba(mitigator, X) -> np.ndarray:
    """
    Approximates predict_proba for mitigator models (e.g., ExponentiatedGradient)
    by computing a weighted average of the base probabilities.
    """
    if not hasattr(mitigator, "predictors_") or not hasattr(mitigator, "weights_"):
        raise ValueError("Mitigator does not expose predictors_ or weights_.")

    probas = np.array([e.predict_proba(X)[:, 1] for e in mitigator.predictors_])
    weights = np.array(mitigator.weights_)
    return np.average(probas, axis=0, weights=weights)


def save_fairness_json(
    path: str | Path,
    group_rows: dict[str, dict[str, float]],
    bias_row: dict[str, float],
    meta: dict[str, float | str],
    overall: dict[str, float] | None = None,
    per_tick: list[dict] | None = None,
    reweight: bool = False,
    group_weights: dict | None = None,
):
    try:

        out = {
            "meta": replace_nans(meta),
            "overall_metrics": replace_nans(overall) if overall else None,
            "by_group": replace_nans(group_rows),
            "bias_metrics": replace_nans(bias_row),
        }
        if per_tick is not None:
            out["per_tick"] = replace_nans(per_tick)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error("Failed to save fairness metrics to JSON: %s", e)


FEATURES = [
    "has_job",
    "has_car",
    "has_house",
    # "wealth",
    # "credit_score",
    # "education",
    "loan_hist",
    "risk_tol",
    "num_children",
    "loan_amount",
    "loan_purpose",
    # "trust",
    "fin_lit",
]


def extract_features(a: Applicant) -> Dict[str, float]:
    """Extracts features from an Applicant into a dictionary"""
    feat_dict = {}
    for key in FEATURES:
        val = getattr(a, key)
        if isinstance(val, bool):
            feat_dict[key] = int(val)
        else:
            feat_dict[key] = float(val)
    return feat_dict


def simulate_abm_ticks(
    rep: float,
    lbl: float,
    seed: int,
    max_ticks: int = MAX_TICKS,
) -> List[Applicant]:
    """Simulates the ABM loop, applying peer influence and decision rules.
    Returns all processed Applicant objects."""

    # reprudicibility
    random.seed(seed)
    np.random.seed(seed)

    agents: Dict[int, Applicant] = {}
    peer_net = nx.Graph()
    processed_agents: List[Applicant] = []

    agent_uid_counter = 0

    for _ in range(max_ticks):
        if random.random() < ARRIVAL_RATE:
            g = "A" if random.random() < rep else "B"
            a = Applicant(agent_uid_counter, g, lbl)

            # if agent trust is high enough, add them to the active agents list and peer network
            if random.random() < a.trust:
                agents[agent_uid_counter] = a
                peer_net.add_node(agent_uid_counter)
                # connect the new agent to up to 3 random existing agents in the peer network
                existing = [
                    peer for peer in peer_net.nodes if peer != agent_uid_counter
                ]
                for peer in random.sample(existing, min(3, len(existing))):
                    peer_net.add_edge(agent_uid_counter, peer)
            agent_uid_counter += 1

        # peer trust + wealth dynamics
        for uid, a in agents.items():
            if peer_net.has_node(uid):
                neighbors = list(peer_net.neighbors(uid))
                if neighbors:
                    neighbor_trusts = [
                        agents[n].trust for n in neighbors if n in agents
                    ]
                    # update agent trust based on the average trust of its neighbors
                    if neighbor_trusts:
                        a.trust = 0.9 * a.trust + 0.1 * np.mean(neighbor_trusts)

                    # wealth transfer between the agent and a random neighbor
                    if random.random() < 0.1:
                        partner = random.choice(neighbors)
                        if partner in agents:
                            b = agents[partner]
                            transfer = 0.05 * min(
                                a.wealth, b.wealth
                            )  # 5% of the smaller wealth
                            a.wealth -= transfer  # deduct transfer amount from agent
                            b.wealth += transfer  # add transfer amount to neighbor

        for uid, a in list(agents.items()):
            # process agents who have been waiting for at least 3 ticks
            if not a.processed and a.waiting_time >= 3:
                # determine if the agent is qualified based on qualification score
                a.qualified = a.qualification_score() >= APPROVAL_THRESHOLD
                # introduce noise to the qualification decision (simulate measuring / data difficulties)
                if random.random() < NOISE_QUAL:
                    a.qualified = not a.qualified
                a.loan_approved = a.qualified
                # apply the effects of the decision on the agent
                a.apply_decision_effect()
                processed_agents.append(a)
            elif not a.processed:
                a.waiting_time += 1
            elif a.processed and a.waiting_time > 15:
                del agents[uid]
            elif a.processed:
                a.waiting_time += 1

    return processed_agents


def _calculate_final_metrics(
    y_all: List[int], probas_all: List[float], groups_all: List[str]
) -> Tuple[Dict, Dict, Dict]:
    """
    Calculates overall, group-specific, and bias metrics from final results.
    """
    if not y_all:
        logging.warning("Calculating metrics with empty results.")
        nan_metrics = {
            k: np.nan
            for k in [
                "roc_auc",
                "log_loss",
                "accuracy",
                "precision",
                "recall",
                "FPR",
                "approval_rate",
            ]
        }
        overall_metrics = {
            k: np.nan
            for k in ["roc_auc", "accuracy", "precision", "recall", "log_loss"]
        }
        group_metrics = {"A": nan_metrics.copy(), "B": nan_metrics.copy()}
        bias_metrics = {k: np.nan for k in ["SPD", "EOD", "AOD", "PrecisionDiff"]}
        return overall_metrics, group_metrics, bias_metrics

    results_df = pd.DataFrame(
        {
            "group": groups_all,
            "actual": y_all,
            "proba": probas_all,
        }
    )
    results_df["predicted"] = (results_df["proba"] >= 0.5).astype(int)

    overall_metrics = {}
    y_true_overall = results_df["actual"]
    y_pred_overall = results_df["predicted"]
    y_proba_overall = results_df["proba"]

    overall_metrics["roc_auc"] = (
        roc_auc_score(y_true_overall, y_proba_overall)
        if len(set(y_true_overall)) > 1
        else np.nan
    )
    overall_metrics["accuracy"] = accuracy_score(y_true_overall, y_pred_overall)
    overall_metrics["precision"] = precision_score(
        y_true_overall, y_pred_overall, zero_division=0
    )
    overall_metrics["recall"] = recall_score(
        y_true_overall, y_pred_overall, zero_division=0
    )
    overall_metrics["log_loss"] = (
        log_loss(y_true_overall, y_proba_overall)
        if len(set(y_true_overall)) > 1
        else np.nan
    )

    group_metrics = {}
    nan_metrics_template = {
        k: np.nan
        for k in [
            "roc_auc",
            "log_loss",
            "accuracy",
            "precision",
            "recall",
            "FPR",
            "approval_rate",
        ]
    }

    for g in ["A", "B"]:
        g_df = results_df[results_df["group"] == g]
        group_metrics[g] = {"support": len(g_df)}
        if g_df.empty:
            group_metrics[g].update(nan_metrics_template.copy())
            continue

        y_true_g, y_pred_g, y_proba_g = g_df["actual"], g_df["predicted"], g_df["proba"]

        tn, fp, fn, tp = 0, 0, 0, 0
        if len(set(y_true_g)) > 1:
            try:
                tn, fp, fn, tp = confusion_matrix(
                    y_true_g, y_pred_g, labels=[0, 1]
                ).ravel()
            except ValueError:
                pass
        elif not y_true_g.empty:  # single class present
            if y_true_g.iloc[0] == 0:
                tn = len(y_true_g)
                fp = 0
            else:
                tp = len(y_true_g)
                fn = 0

        group_metrics[g]["roc_auc"] = (
            roc_auc_score(y_true_g, y_proba_g) if len(set(y_true_g)) > 1 else np.nan
        )
        group_metrics[g]["log_loss"] = (
            log_loss(y_true_g, y_proba_g) if len(set(y_true_g)) > 1 else np.nan
        )
        group_metrics[g]["accuracy"] = accuracy_score(y_true_g, y_pred_g)
        group_metrics[g]["precision"] = precision_score(
            y_true_g, y_pred_g, zero_division=0
        )
        group_metrics[g]["recall"] = recall_score(
            y_true_g, y_pred_g, zero_division=0
        )  # TPR
        group_metrics[g]["FPR"] = (
            fp / (fp + tn) if (fp + tn) > 0 else 0.0
        )  # 0.0 instead of NaN if denominator is 0
        group_metrics[g]["approval_rate"] = (
            y_pred_g.mean() if not y_pred_g.empty else np.nan
        )

    gA_metrics, gB_metrics = group_metrics["A"], group_metrics["B"]
    bias_metrics = {
        "SPD": gB_metrics.get("approval_rate", np.nan)
        - gA_metrics.get("approval_rate", np.nan),
        "EOD": gB_metrics.get("recall", np.nan) - gA_metrics.get("recall", np.nan),
        "AOD": 0.5
        * (
            abs(gA_metrics.get("recall", np.nan) - gB_metrics.get("recall", np.nan))
            + abs(gA_metrics.get("FPR", np.nan) - gB_metrics.get("FPR", np.nan))
        ),
        "PrecisionDiff": gB_metrics.get("precision", np.nan)
        - gA_metrics.get("precision", np.nan),
    }

    return overall_metrics, group_metrics, bias_metrics


def _process_agents_online(
    model,
    agents,
    calibrate_interval,
    run_explainability=False,
    xai_tag="",
    X_train_for_xai=None,
    run_XAI=False,
):
    """
    Processes a list of agents using an online model, updating it and collecting metrics.
    """
    cm = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    total_seen = {"A": 0, "B": 0}
    total_approved = {"A": 0, "B": 0}
    probas_all, X_all_dicts, X_test_buffer_dicts, y_all, groups_all = [], [], [], [], []
    metrics_per_tick = []

    logging.info(
        f"Starting agent processing loop {'with XAI' if run_explainability else 'for tuning/evaluation'}."
    )
    for tick, a in enumerate(agents):
        if not a.processed:
            continue

        x_dict = extract_features(a)
        X_all_dicts.append(x_dict)
        X_test_buffer_dicts.append(x_dict)  # add current features to test (xai) buffer

        # constrain buffer size to the calibrate_interval for XAI
        if len(X_test_buffer_dicts) > calibrate_interval:
            X_test_buffer_dicts.pop(0)

        y_true = int(a.qualified)
        group = a.group

        x_df = pd.DataFrame([x_dict])
        proba = model.predict_proba(x_df)[0, 1]
        # pred = int(proba >= 0.5)
        # use computed threshold
        pred_arr = model.predict(x_df)
        pred = pred_arr[0, 1] if pred_arr.ndim == 2 else pred_arr[0]

        model.update(x_df, pd.Series([y_true]), pd.Series([group]))

        probas_all.append(proba)
        y_all.append(y_true)
        groups_all.append(group)

        if pred == 1 and y_true == 1:
            cm[group]["TP"] += 1
        elif pred == 1 and y_true == 0:
            cm[group]["FP"] += 1
        elif pred == 0 and y_true == 1:
            cm[group]["FN"] += 1
        else:
            cm[group]["TN"] += 1

        if pred == 1:
            total_approved[group] += 1
        total_seen[group] += 1

        # store metrics periodically (e.g., every 10 ticks)
        if tick % 10 == 0:
            appr_rate_A = (
                total_approved["A"] / total_seen["A"] if total_seen["A"] > 0 else np.nan
            )
            appr_rate_B = (
                total_approved["B"] / total_seen["B"] if total_seen["B"] > 0 else np.nan
            )
            tpr_A = (
                cm["A"]["TP"] / (cm["A"]["TP"] + cm["A"]["FN"])
                if (cm["A"]["TP"] + cm["A"]["FN"]) > 0
                else np.nan
            )
            tpr_B = (
                cm["B"]["TP"] / (cm["B"]["TP"] + cm["B"]["FN"])
                if (cm["B"]["TP"] + cm["B"]["FN"]) > 0
                else np.nan
            )
            metrics_per_tick.append(
                {
                    "tick": tick,
                    "approval_rate_a": appr_rate_A,
                    "approval_rate_b": appr_rate_B,
                    "dp_gap": (
                        abs(appr_rate_A - appr_rate_B)
                        if not (np.isnan(appr_rate_A) or np.isnan(appr_rate_B))
                        else np.nan
                    ),
                    "tpr_a": tpr_A,
                    "tpr_b": tpr_B,
                    "eo_gap": (
                        abs(tpr_A - tpr_B)
                        if not (np.isnan(tpr_A) or np.isnan(tpr_B))
                        else np.nan
                    ),
                }
            )

    if run_explainability and run_XAI:
        # explainability (XAI)
        final_model_state = model.mitigator_model if model.use_mitigator else model
        if (
            X_train_for_xai is not None
            and not X_train_for_xai.empty
            and X_test_buffer_dicts
        ):
            logging.info(f"Running explainability analysis with tag: {xai_tag}")
            X_test_df_for_xai = pd.DataFrame(X_test_buffer_dicts, columns=FEATURES)
            explainability_analysis(
                model=final_model_state,
                X_train=X_train_for_xai,  # full history DataFrame
                X_test=X_test_df_for_xai,  # last batch DataFrame
                output_dir=XAI_DIR,
                prefix=xai_tag,
            )
        else:
            logging.warning("Skipping explainability.")

    results = {
        "metrics_per_tick": metrics_per_tick,
        "probas_all": probas_all,
        "y_all": y_all,
        "groups_all": groups_all,
        "X_all_dicts": X_all_dicts,
        "X_test_buffer_dicts": X_test_buffer_dicts,
    }

    return results


def simulate_dynamic_loop(model, rep, lbl, seed):
    """
    Runs a simulation loop for tuning.
    """
    agents = simulate_abm_ticks(rep, lbl, seed, max_ticks=MAX_TICKS)

    logging.info("Running dynamic loop for tuning online model...")
    processed_results = _process_agents_online(
        model=model,
        agents=agents,
        calibrate_interval=model.calibrate_interval,
        run_explainability=False,
    )

    overall_metrics, group_metrics, bias_metrics = _calculate_final_metrics(
        y_all=processed_results["y_all"],
        probas_all=processed_results["probas_all"],
        groups_all=processed_results["groups_all"],
    )

    # metrics for hyperparameter tuning
    return (
        processed_results["metrics_per_tick"],
        overall_metrics,
        group_metrics,
        bias_metrics,
    )


def run_dynamic(
    rep: float,
    lbl: float,
    setting: str,
    update_interval: int,
    seed: int,
    reweight: bool = False,
    group_weights: dict | None = None,
    calibrate_interval: int = 100,
    use_mitigator: bool = False,
    fairness_constraint: str = "demographic_parity",
    run_XAI: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Runs the dynamic online learning pipeline.
    """
    CSV_DIR, JSON_DIR, XAI_DIR = get_dirs(setting)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(XAI_DIR, exist_ok=True)

    np.random.seed(seed)
    random.seed(seed)

    logging.info(
        "Starting dynamic run: rep=%.2f, lbl=%.2f, seed=%d, update_interval=%d, calibrate_interval=%d, reweight=%s, mitigator=%s (%s)",
        rep,
        lbl,
        seed,
        update_interval,
        calibrate_interval,
        reweight,
        use_mitigator,
        fairness_constraint,
    )

    tune_tree = not use_mitigator
    if not tune_tree:
        logging.info(
            "Tree hyper‑parameter tuning skipped " "(tune_tree=%s, use_mitigator=%s).",
            tune_tree,
            use_mitigator,
        )
    if tune_tree:
        param_grid = {
            "grace_period": [20, 50, 100, 200],
            "delta": [1e-4, 1e-5, 1e-6],
            "max_depth": [5, 10, 20, None],
            "leaf_prediction": ["mc", "nb", "nba"],
            "split_criterion": ["info_gain", "gini", "hellinger"],
            "tau": [0.01, 0.05, 0.1],
            "nb_threshold": [0, 10, 30],
            "binary_split": [False, True],
            "remove_poor_attrs": [False, True],
            "merit_preprune": [True, False],
        }
        # For consistency with RandomizedSearchCV (on static, which performs 3 iterations), n_iter is set to 3.
        sampled_params = list(ParameterSampler(param_grid, n_iter=3, random_state=seed))
        best_score = -np.inf
        best_params = None

        logging.info("Starting hyperparameter tuning for online model...")
        for params in sampled_params:
            tuning_model = OnlineApprovalClassifier(
                update_interval=update_interval,
                calibrate_interval=calibrate_interval,
                reweight=reweight,
                group_weights=group_weights,
                use_mitigator=use_mitigator,
                fairness_constraint=fairness_constraint,
                **params,
            )
            _, overall_metrics_tune, _, _ = simulate_dynamic_loop(
                tuning_model, rep, lbl, seed
            )
            score = overall_metrics_tune.get("roc_auc", 0)
            if np.isnan(score):
                score = 0
            logging.debug(f"Tuning run params: {params}, Score (ROC AUC): {score:.4f}")
            if score > best_score:
                best_score = score
                best_params = params
                logging.debug(f"Found new best score: {best_score:.4f}")

        if best_params is None:
            logging.warning(
                "Hyperparameter tuning did not find best parameters. Using the first sampled parameters as fallback."
            )
            best_params = sampled_params[0] if sampled_params else {}

        logging.info(f"Hyperparameter tuning complete. Best ROC AUC: {best_score:.4f}")
        logging.info(f"Best online parameters found: {best_params}")
    else:
        best_params = {}

    final_model = OnlineApprovalClassifier(
        update_interval=update_interval,
        calibrate_interval=calibrate_interval,
        reweight=reweight,
        group_weights=group_weights,
        use_mitigator=use_mitigator,
        fairness_constraint=fairness_constraint,
        **best_params,
    )

    logging.info("Generating final agent data stream using simulate_abm_ticks...")
    final_agents = simulate_abm_ticks(rep, lbl, seed, max_ticks=MAX_TICKS)

    all_processed_agent_data = [
        extract_features(a)
        | {"id": a.id, "qualified": int(a.qualified), "group": a.group}
        for a in final_agents
        if a.processed
    ]
    df_final_agents_full = pd.DataFrame(all_processed_agent_data)

    if not df_final_agents_full.empty:
        X_full_df_for_xai = df_final_agents_full[FEATURES]
    else:
        X_full_df_for_xai = pd.DataFrame(columns=FEATURES)

    if reweight:
        if group_weights is not None:
            group_weights_suffix = "_".join(
                f"{k}{v}" for k, v in sorted(group_weights.items())
            )
            reweight_suffix = f"_reweight_manual_{group_weights_suffix}"
        else:
            reweight_suffix = "_reweight_auto"
    else:
        reweight_suffix = ""
    tag_base = (
        f"dynamic_rep{rep}_lbl{lbl}_upd{update_interval}{reweight_suffix}_seed{seed}"
    )
    if use_mitigator:
        tag_base += "_mitigator_" + fairness_constraint

    csv_path = Path(CSV_DIR) / f"abm_{tag_base}.csv"
    if not df_final_agents_full.empty:
        df_final_agents_full.to_csv(csv_path, index=False)
        logging.info(f"Saved generated agent data to {csv_path}")
    else:
        logging.warning(
            "No processed agents generated for the final run. Skipping CSV save."
        )

    xai_tag = f"dynamic_rep{rep}_lbl{lbl}_upd{update_interval}_seed{seed}"
    if reweight:
        xai_tag += reweight_suffix
    if group_weights:
        xai_tag += "_manual_weights_" + "_".join(
            f"{k}{v}" for k, v in sorted(group_weights.items())
        )
    if use_mitigator:
        xai_tag += "_mitigator_" + fairness_constraint

    logging.info("Starting final online processing loop with best model...")
    final_processed_results = _process_agents_online(
        model=final_model,
        agents=final_agents,
        calibrate_interval=calibrate_interval,
        run_explainability=True,
        xai_tag=xai_tag,
        X_train_for_xai=X_full_df_for_xai,
        run_XAI=run_XAI,
    )

    overall_metrics_final, group_metrics_final, bias_metrics_final = (
        _calculate_final_metrics(
            y_all=final_processed_results["y_all"],
            probas_all=final_processed_results["probas_all"],
            groups_all=final_processed_results["groups_all"],
        )
    )

    meta = {
        "mode": setting,
        "update_interval": update_interval,
        "calibrate_interval": calibrate_interval,
        "representation_bias": rep,
        "label_bias": lbl,
        "max_ticks": MAX_TICKS,
        "seed": seed,
        "reweighting_enabled": reweight,
        "reweighting_mode": (
            "manual" if group_weights else "auto" if reweight else "none"
        ),
        "manual_group_weights": group_weights if group_weights else None,
        "best_online_params": best_params,
        "mitigator_enabled": use_mitigator,
        "mitigator_constraint": fairness_constraint if use_mitigator else None,
    }

    json_path = Path(JSON_DIR) / f"metrics_{tag_base}.json"
    save_fairness_json(
        path=json_path,
        group_rows=group_metrics_final,
        bias_row=bias_metrics_final,
        meta=meta,
        overall=overall_metrics_final,
        per_tick=final_processed_results["metrics_per_tick"],
        reweight=reweight,
        group_weights=group_weights,
    )
    logging.info(f"Saved final metrics and metadata to {json_path}")

    return (
        pd.DataFrame(final_processed_results["metrics_per_tick"]),
        overall_metrics_final,
    )


def run_static(
    rep: float,
    lbl: float,
    seed: int,
    reweight: bool = False,
    group_weights: dict | None = None,
    use_mitigator: bool = False,
    fairness_constraint: str = "demographic_parity",
    run_XAI: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Runs a static pipeline: Generate data -> Train XGBoost -> Calibrate -> Evaluate.
    """
    CSV_DIR, JSON_DIR, XAI_DIR = get_dirs("static")
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(XAI_DIR, exist_ok=True)

    logging.info(
        "Starting static run: rep=%.2f, lbl=%.2f, reweight=%s, group_weights=%s, seed=%d",
        rep,
        lbl,
        reweight,
        group_weights,
        seed,
    )

    agents = simulate_abm_ticks(rep, lbl, seed, max_ticks=MAX_TICKS)
    df = pd.DataFrame(
        [
            extract_features(a)
            | {"id": a.id, "qualified": int(a.qualified), "group": a.group}
            for a in agents
            if a.processed
        ]
    )

    if df.empty:
        logging.error("Static data generation failed.")
        return pd.DataFrame(), {}

    X = df[FEATURES]
    y = df["qualified"]
    groups = df["group"]

    # (60 / 20 / 20)
    X_train_cal, X_test, y_train_cal, y_test, g_train_cal, g_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_cal, y_train, y_cal, g_train, g_cal = train_test_split(
        X_train_cal,
        y_train_cal,
        g_train_cal,
        test_size=0.2,
        random_state=seed,
        stratify=y_train_cal,
    )
    logging.info(
        "Data split: Train=%d, Calibrate=%d, Test=%d",
        len(y_train),
        len(y_cal),
        len(y_test),
    )

    scaler = SklearnStandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_cal = pd.DataFrame(scaler.transform(X_cal), columns=FEATURES, index=X_cal.index)
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURES, index=X_test.index
    )

    w_train, w_cal = None, None
    reweighting_mode = "none"
    if reweight:
        if group_weights is not None:
            logging.info("Applying manual group weights: %s", group_weights)
            w_train = g_train.map(group_weights).fillna(1.0).to_numpy()
            w_cal = g_cal.map(group_weights).fillna(1.0).to_numpy()
            reweighting_mode = "manual"
            logging.info("Manual weights for training: %s", list(zip(g_train, w_train)))
            logging.info("Manual weights for calibration: %s", list(zip(g_cal, w_cal)))

            group_weights_suffix = "_".join(
                f"{k}{v}" for k, v in sorted(group_weights.items())
            )
            reweight_suffix = f"_reweight_manual_{group_weights_suffix}"
        else:
            logging.info("Applying automatic reweighing using AIF360 logic.")
            try:
                X_train_G = X_train.copy()
                X_train_G["group"] = g_train
                X_train_G = X_train_G.rename(columns=lambda c: c.strip())
                assert "group" in X_train_G.columns, list(X_train_G.columns)
                rw_t = Reweighing(prot_attr=g_train)
                _, w_train = rw_t.fit_transform(X_train_G, y_train)
                logging.info(
                    "Auto weights for training: %s", list(zip(g_train, w_train))
                )

                X_cal_G = X_cal.copy()
                X_cal_G["group"] = g_cal
                X_cal_G = X_cal_G.rename(columns=lambda c: c.strip())
                assert "group" in X_cal_G.columns, list(X_cal_G.columns)
                rw_c = Reweighing(prot_attr=g_cal)
                _, w_cal = rw_c.fit_transform(X_cal_G, y_cal)
                logging.info(
                    "Auto weights for calibration: %s", list(zip(g_cal, w_cal))
                )

                reweighting_mode = "auto"
                reweight_suffix = "_reweight_auto"
                group_weights = {
                    "A": w_train[g_train == "A"].mean(),
                    "B": w_train[g_train == "B"].mean(),
                }
                logging.info("Auto weights for training: %s", group_weights)
            except Exception as e:
                logging.error(
                    "Automatic reweighing failed: %s. Proceeding without weights.", e
                )
                reweight = False
                reweighting_mode = "error"
                reweight_suffix = ""
    else:
        reweight_suffix = ""

    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.5, 1],
        "min_child_weight": [1, 3, 5],
    }

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=seed,
    )

    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        scoring="roc_auc",
        n_jobs=-1,
        n_iter=3,
        random_state=seed,
    )

    logging.info(
        "Starting XGBoost training with RandomizedSearch%s...",
        " and sample weights" if reweight else "",
    )
    fit_params = {"sample_weight": w_train} if reweight and w_train is not None else {}
    random_search.fit(X_train, y_train, **fit_params)

    if not reweight and use_mitigator:
        logging.info(
            "Training XGBoost using ExponentiatedGradient (DemographicParity)."
        )
        base_estimator = random_search.best_estimator_

        if fairness_constraint == "demographic_parity":
            constraint = DemographicParity()
        elif fairness_constraint == "equalized_odds":
            constraint = EqualizedOdds()
        else:
            raise ValueError(f"Unsupported constraint: {fairness_constraint}")

        mitigator = ExponentiatedGradient(base_estimator, constraints=constraint)
        mitigator.fit(X_train, y_train, sensitive_features=g_train)
        best_xgb = mitigator
        logging.info("Mitigator training complete.")
    else:
        best_xgb = random_search.best_estimator_

    logging.info(
        "XGBoost training complete. Best score: %.4f", random_search.best_score_
    )
    logging.info("Best params: %s", random_search.best_params_)

    if not reweight and use_mitigator:
        # mitigator does not support predicting probabilities
        logging.info("Skipping calibration for mitigator.")
        calibrator = best_xgb
    else:
        logging.info(
            "Starting Isotonic Calibration%s...",
            " with sample weights" if reweight else "",
        )
        calibrator = CalibratedClassifierCV(best_xgb, method="isotonic", cv="prefit")
        fit_params_cal = (
            {"sample_weight": w_cal} if reweight and w_cal is not None else {}
        )
        try:
            calibrator.fit(X_cal, y_cal, **fit_params_cal)
            logging.info("Isotonic calibration complete.")
        except Exception as e:
            logging.error(
                "Isotonic calibration failed: %s. Proceeding without calibration.", e
            )
            calibrator = best_xgb

    logging.info("Evaluating on test set...")
    try:
        proba_test = calibrator.predict_proba(X_test)[:, 1]
    except AttributeError:
        if hasattr(calibrator, "predictors_") and hasattr(calibrator, "weights_"):
            logging.info("Estimating proba_test from ExponentiatedGradient ensemble.")
            proba_test = ensemble_predict_proba(calibrator, X_test)
        else:
            logging.error(
                "Model has no predict_proba or predictors_; cannot compute probabilities."
            )
            raise

    overall_metrics, group_metrics, bias_metrics = _calculate_final_metrics(
        y_all=y_test.tolist(),
        probas_all=proba_test.tolist(),
        groups_all=g_test.tolist(),
    )

    if run_XAI:
        final_model = calibrator if calibrator is not None else best_xgb
        logging.info("Final model: %s", final_model.__class__.__name__)
        xai_tag = f"static_rep{rep}_lbl{lbl}_seed{seed}"
        if reweight:
            xai_tag += reweight_suffix
        if group_weights:
            xai_tag += "_manual_weights_" + "_".join(
                f"{k}{v}" for k, v in sorted(group_weights.items())
            )
        if use_mitigator:
            xai_tag += "_mitigator_" + fairness_constraint

        explainability_analysis(
            model=final_model,
            X_train=X_train,
            X_test=X_test,
            output_dir=XAI_DIR,
            prefix=xai_tag,
        )

    meta = {
        "mode": "static",
        "representation_bias": rep,
        "label_bias": lbl,
        "seed": seed,
        "reweighting_enabled": reweight,
        "reweighting_mode": reweighting_mode,
        "manual_group_weights": group_weights if group_weights else None,
        "model": "XGBoost",
        "calibration": (
            "Isotonic" if isinstance(calibrator, CalibratedClassifierCV) else "None"
        ),
    }

    tag = f"static_rep{rep}_lbl{lbl}{reweight_suffix}_seed{seed}"
    if use_mitigator:
        tag += "_mitigator_" + fairness_constraint

    json_path = Path(JSON_DIR) / f"metrics_{tag}.json"
    save_fairness_json(
        json_path,
        group_metrics,
        bias_metrics,
        meta,
        overall=overall_metrics,
        reweight=reweight,
        group_weights=group_weights,
    )

    csv_path = (
        Path(CSV_DIR) / f"abm_static_run_{rep}_lbl{lbl}_seed{seed}{reweight_suffix}.csv"
    )
    if use_mitigator:
        csv_path = csv_path.with_name(csv_path.stem + "_mitigator" + csv_path.suffix)
    df.to_csv(csv_path, index=False)

    return pd.DataFrame(), overall_metrics


def run_experiment(
    rep: float,
    lbl: float,
    seed: int,
    setting: str,
    reweight: bool = False,
    group_weights: dict | None = None,
    update_interval: int = 10,
    calibrate_interval: int = 100,
    use_mitigator: bool = False,
    fairness_constraint: str = "demographic_parity",
    XAI: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    # --- Early skip if output JSON already exists ---
    from pathlib import Path
    import logging

    # Determine output JSON path using the same logic as at the end of run_dynamic/run_static
    if setting == "dynamic":
        _, JSON_DIR, _ = get_dirs(setting)
        if reweight:
            if group_weights is not None:
                group_weights_suffix = "_".join(
                    f"{k}{v}" for k, v in sorted(group_weights.items())
                )
                reweight_suffix = f"_reweight_manual_{group_weights_suffix}"
            else:
                reweight_suffix = "_reweight_auto"
        else:
            reweight_suffix = ""
        tag_base = f"dynamic_rep{rep}_lbl{lbl}_upd{update_interval}{reweight_suffix}_seed{seed}"
        if use_mitigator:
            tag_base += "_mitigator_" + fairness_constraint
        json_path = Path(JSON_DIR) / f"metrics_{tag_base}.json"
    elif setting == "static":
        _, JSON_DIR, _ = get_dirs("static")
        if reweight:
            if group_weights is not None:
                group_weights_suffix = "_".join(
                    f"{k}{v}" for k, v in sorted(group_weights.items())
                )
                reweight_suffix = f"_reweight_manual_{group_weights_suffix}"
            else:
                reweight_suffix = "_reweight_auto"
        else:
            reweight_suffix = ""
        tag = f"static_rep{rep}_lbl{lbl}{reweight_suffix}_seed{seed}"
        if use_mitigator:
            tag += "_mitigator_" + fairness_constraint
        json_path = Path(JSON_DIR) / f"metrics_{tag}.json"
    else:
        raise ValueError("Unsupported mode. Use 'static' or 'dynamic'.")

    if json_path.exists():
        logging.info(f"Skipping run: output already exists at {json_path}")
        return pd.DataFrame(), {}

    if setting == "dynamic":
        return run_dynamic(
            rep=rep,
            lbl=lbl,
            setting=setting,
            update_interval=update_interval,
            seed=seed,
            reweight=reweight,
            group_weights=group_weights,
            calibrate_interval=calibrate_interval,
            use_mitigator=use_mitigator,
            fairness_constraint=fairness_constraint,
            run_XAI=XAI,
        )
    elif setting == "static":
        return run_static(
            rep=rep,
            lbl=lbl,
            seed=seed,
            reweight=reweight,
            group_weights=group_weights,
            use_mitigator=use_mitigator,
            fairness_constraint=fairness_constraint,
            run_XAI=XAI,
        )
    else:
        raise ValueError("Unsupported mode. Use 'static' or 'dynamic'.")


def main():
    ap = argparse.ArgumentParser(
        description="Run Agent-Based Model simulation with static or online learning."
    )
    ap.add_argument("--setting", choices=["static", "dynamic"], required=True)
    ap.add_argument("--rep", type=float, default=0.5)
    ap.add_argument("--lbl", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--update-interval", type=int, default=int(MAX_TICKS * 0.01))
    ap.add_argument("--calibrate-interval", type=int, default=int(MAX_TICKS * 0.1))
    ap.add_argument("--reweighting", action="store_true")
    ap.add_argument("--group-weights", type=str, default=None)
    ap.add_argument("--mitigator", action="store_true")
    ap.add_argument(
        "--fairness-constraint",
        choices=["demographic_parity", "equalized_odds"],
        default="demographic_parity",
        help="Fairness constraint to apply if --mitigator is used.",
    )
    ap.add_argument("--xai", action="store_true")
    ap.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Script arguments: %s", args)

    manual_weights = None
    if args.group_weights:
        try:
            manual_weights = json.loads(args.group_weights)
            if not isinstance(manual_weights, dict):
                raise ValueError("Group weights must be a JSON dictionary.")
            logging.info("Parsed manual group weights: %s", manual_weights)
        except Exception as e:
            logging.error(
                "Failed to parse --group-weights JSON string: %s. Disabling manual weights.",
                e,
            )
            manual_weights = None

    run_experiment(
        rep=args.rep,
        lbl=args.lbl,
        seed=args.seed,
        reweight=args.reweighting,
        setting=args.setting,
        group_weights=manual_weights,
        update_interval=args.update_interval,
        calibrate_interval=args.calibrate_interval,
        use_mitigator=args.mitigator,
        fairness_constraint=args.fairness_constraint,
        XAI=args.xai,
    )


if __name__ == "__main__":
    main()

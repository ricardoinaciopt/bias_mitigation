import logging
from typing import Any, Dict, List
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from river import tree, preprocessing as river_preprocessing
from xgboost import XGBClassifier as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import IsotonicRegression
from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity,
    EqualizedOdds,
)


class StabilizedOnlineReweighter:
    def __init__(self, alpha=0.001, warmup=200, clip_min=0.1, clip_max=5.0):
        self.alpha = alpha
        self.warmup = warmup
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.count = 0
        self.ema_total = 0.0
        self.ema_y = defaultdict(float)
        self.ema_g = defaultdict(float)
        self.ema_gy = defaultdict(float)

    def update(self, g, y):
        self.count += 1
        # update total EMA
        self.ema_total = (1 - self.alpha) * self.ema_total + self.alpha
        # update class EMA
        self.ema_y[y] = (1 - self.alpha) * self.ema_y[y] + self.alpha
        # update group EMA
        self.ema_g[g] = (1 - self.alpha) * self.ema_g[g] + self.alpha
        # update joint EMA
        self.ema_gy[(g, y)] = (1 - self.alpha) * self.ema_gy[(g, y)] + self.alpha

    def weight(self, g, y):
        # no weighting during warm-up
        if self.count < self.warmup or self.ema_gy[(g, y)] == 0:
            return 1.0

        p_y = self.ema_y[y] / self.ema_total
        p_g = self.ema_g[g] / self.ema_total
        p_gy = self.ema_gy[(g, y)] / self.ema_total

        w = (p_y * p_g) / (p_gy)
        return float(np.clip(w, self.clip_min, self.clip_max))


class OnlineApprovalClassifier:
    """
    Incremental Hoeffding Tree classifier with online sample reweightting
    and optional periodic isotonic calibration.
    """

    def __init__(
        self,
        update_interval: int = 10,
        calibrate_interval: int = 100,
        retrain_interval: int = None,
        reweight: bool = False,
        group_weights: dict[str, float] | None = None,
        use_mitigator: bool = False,
        fairness_constraint: str = "demographic_parity",
        **model_params,
    ):

        self.update_interval = max(1, update_interval)
        self.calibrate_interval = max(self.update_interval, calibrate_interval)
        self.retrain_interval = retrain_interval or self.calibrate_interval
        self.reweight = reweight
        self.group_weights = group_weights
        self.use_mitigator = use_mitigator
        self.fairness_constraint = fairness_constraint
        self.mitigator_model = None
        self._mitigator_fitted = False

        self.tree = tree.HoeffdingTreeClassifier(**model_params)
        self.scaler = river_preprocessing.StandardScaler()
        self.iso = None
        self._calX_buffer: List[float] = (
            []
        )  # buffer wiht raw probabilities for calibration
        self._caly_buffer: List[int] = []  # buffer with true labels for calibration
        self._interval_reached = False

        # training buffers (hold data until self.update_interval is reached)
        self._bufX: List[Dict] = []
        self._bufy: List[int] = []
        self._bufG: List[str | Any] = []

        if self.reweight and self.group_weights is None:
            self.rw = StabilizedOnlineReweighter()
            logging.info("Online automatic reweighting enabled.")
        elif self.reweight and self.group_weights:
            self.rw = None
            logging.info("Manual group reweighting enabled: %s", self.group_weights)
        else:
            self.rw = None
            logging.info("reweighting disabled.")

        self._samples_seen_since_cal = 0
        self._samples_seen_since_retrain = 0

        # buffers for rolling‑window mitigator
        self._mitBufferX = deque(maxlen=self.calibrate_interval)
        self._mitBuffery = deque(maxlen=self.calibrate_interval)
        self._mitBufferg = deque(maxlen=self.calibrate_interval)

    def _get_weight(self, g: str | Any, y: int) -> float:
        """Calculates the sample weight based on the active reweighting strategy."""
        if not self.reweight:
            return 1.0
        if self.group_weights:
            return self.group_weights.get(g, 1.0)
        if self.rw:
            self.rw.update(g, y)
            return self.rw.weight(g, y)
        return 1.0

    def _learn_one(self, x: dict, y: int, g: str | Any):
        """Process a single instance: scale, get weight, learn."""
        w = self._get_weight(g, y)

        self.scaler.learn_one(x)
        x_scaled = self.scaler.transform_one(x)
        if not x_scaled:
            return
        self.tree.learn_one(x_scaled, y, w=w)

        if self.use_mitigator:
            self._mitBufferX.append(x_scaled)
            self._mitBuffery.append(y)
            self._mitBufferg.append(g)

            self._samples_seen_since_retrain = 1

    def _raw_proba_one(self, x: dict) -> float:
        """Get the raw probability estimate from the tree for positive classifications."""
        try:
            x_scaled = self.scaler.transform_one(x)
            if not x_scaled:
                return 0.5
        except Exception:
            return 0.5

        probas = self.tree.predict_proba_one(x_scaled)

        return probas.get(1, 0.5)

    def _mitigator_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Reconstruct probabilistic output from ExponentiatedGradient ensemble."""
        if not hasattr(self.mitigator_model, "predictors_") or not hasattr(
            self.mitigator_model, "weights_"
        ):
            logging.warning(
                "Mitigator lacks ensemble structure; returning 0.5 for all."
            )
            return np.full(len(X), 0.5)

        probas = np.array(
            [est.predict_proba(X)[:, 1] for est in self.mitigator_model.predictors_]
        )
        weights = np.array(self.mitigator_model.weights_)
        return np.average(probas, axis=0, weights=weights)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # mitigator branch
        if self.use_mitigator and self.mitigator_model is not None:
            if hasattr(self.mitigator_model, "predict_proba"):
                try:
                    pm = self.mitigator_model.predict_proba(X)
                    # already 2-column? just return
                    if pm.ndim == 2 and pm.shape[1] == 2:
                        return pm
                    # else assume it's a vector of P(1)
                    p1 = pm.ravel()
                    p0 = 1.0 - p1
                    return np.column_stack((p0, p1))
                except Exception as e:
                    logging.warning(
                        "mitigator.predict_proba failed (%s); falling back to ensemble helper",
                        e,
                    )
            # fallback
            p1 = self._mitigator_proba(X)
            p0 = 1.0 - p1
            return np.column_stack((p0, p1))

        # raw / calibration branch
        raw_probabilities = np.array(
            [self._raw_proba_one(row.to_dict()) for _, row in X.iterrows()]
        )
        proba_1 = raw_probabilities
        if self.iso is not None:
            clipped = np.clip(proba_1, 1e-6, 1.0 - 1e-6)
            try:
                proba_1 = self.iso.transform(clipped)
            except Exception as e:
                logging.warning(
                    "Isotonic transform failed: %s; using raw probabilities.", e
                )
        proba_0 = 1.0 - proba_1
        return np.column_stack((proba_0, proba_1))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels (0 or 1) for a batch of instances."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if X.empty:
            return np.array([])
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def update(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
        """Buffers incoming data and triggers learning/calibration updates."""
        if X.empty:
            return

        self._bufX.extend(X.to_dict(orient="records"))
        self._bufy.extend(y.tolist())
        self._bufG.extend(groups.tolist())

        if len(self._bufy) >= self.update_interval:
            self._flush_buffers()

    def _flush_buffers(self):
        """Learn from buffered data and manage calibration."""
        if not self._bufy:
            return
        logging.debug("Flushing buffers. Size: %d", len(self._bufy))

        for x_, y_, g_ in zip(self._bufX, self._bufy, self._bufG):
            self._learn_one(x_, y_, g_)

        # raw probabilities and true labels from this flushed batch -> calibration buffer
        current_raw_probs = [self._raw_proba_one(x_) for x_ in self._bufX]
        self._calX_buffer.extend(current_raw_probs)
        self._caly_buffer.extend(self._bufy)
        self._samples_seen_since_cal += len(self._bufy)

        if self._samples_seen_since_cal >= self.calibrate_interval:
            self._interval_reached = True

        if self._interval_reached:
            if self.use_mitigator and (
                not self._mitigator_fitted
                or self._samples_seen_since_retrain >= self.retrain_interval
            ):
                self._fit_mitigator_rolling()
                self._mitigator_fitted = True
                self._samples_seen_since_retrain = 0
                self._mitigator_fitted = True
            elif not self.use_mitigator:
                self._fit_calibration()
            self._interval_reached = False

        self._bufX.clear()
        self._bufy.clear()
        self._bufG.clear()

    def _fit_calibration(self):
        """Fits the Isotonic Regression model using the calibration buffer."""
        if not self._caly_buffer:
            logging.warning("Calibration buffer empty, cannot fit IsotonicRegression.")
            return

        cal_X_arr = np.array(self._calX_buffer)
        cal_y_arr = np.array(self._caly_buffer)

        finite_mask = np.isfinite(cal_X_arr) & np.isfinite(cal_y_arr)
        unique_classes = len(np.unique(cal_y_arr[finite_mask]))

        if np.sum(finite_mask) >= 10 and unique_classes > 1:
            logging.info(
                "Fitting IsotonicRegression on %d points.", np.sum(finite_mask)
            )
            try:
                self.iso = IsotonicRegression(out_of_bounds="clip")
                self.iso.fit(cal_X_arr[finite_mask], cal_y_arr[finite_mask])
                self._calX_buffer.clear()
                self._caly_buffer.clear()
                self._samples_seen_since_cal = 0
            except Exception as e:
                logging.error("Failed to fit IsotonicRegression: %s", e)
                self.iso = None
        else:
            logging.warning(
                "Skipping IsotonicRegression fit: Need >= 10 finite points and > 1 class. "
                "Points: %d, Classes: %d",
                np.sum(finite_mask),
                unique_classes,
            )

    def _fit_mitigator_rolling(self):
        logging.info(
            "Rolling-window mitigator fit on last %d samples.",
            len(self._mitBuffery),
        )
        if (
            len(self._mitBuffery) < self.calibrate_interval // 2
            or len(set(self._mitBuffery)) < 2
        ):
            logging.warning("Not enough data or class diversity; skipping fit.")
            return

        X_df = pd.DataFrame(list(self._mitBufferX))
        y = np.fromiter(self._mitBuffery, dtype=int)
        g = np.fromiter(self._mitBufferg, dtype=object)

        if hasattr(self, "_best_xgb_params"):
            base_model = xgb(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                **self._best_xgb_params,
            )
            base_model.set_params(process_type="update", refresh_leaf=True)
        else:
            base_model = xgb(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )

        if not hasattr(self, "_best_xgb_params"):
            param_dist = {
                "n_estimators": [100, 200, 300, 400],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.5, 1],
                "min_child_weight": [1, 3, 5],
            }
            base_model = RandomizedSearchCV(
                base_model, param_distributions=param_dist, n_jobs=-1, random_state=42
            )
            base_model.fit(X_df, y)
            self._best_xgb_params = base_model.best_params_
            best_est = base_model.best_estimator_
        else:
            best_est = base_model.fit(X_df, y)

        constraint = (
            EqualizedOdds()
            if self.fairness_constraint == "equalized_odds"
            else DemographicParity()
        )
        mitigator = ExponentiatedGradient(
            estimator=best_est, constraints=constraint, eps=0.01
        )
        mitigator.fit(X_df, y, sensitive_features=g)
        self.mitigator_model = mitigator

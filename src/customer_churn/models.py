from __future__ import annotations

from typing import Any


def model_zoo() -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    zoo: dict[str, Any] = {
        # Interpretable floor; a retention team can read the coefficients.
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        # The notebook's model, reproduced exactly for comparability.
        "random_forest_notebook": RandomForestClassifier(n_estimators=100, random_state=42),
        # The same model told that the classes are imbalanced.
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=400, random_state=42, class_weight="balanced_subsample", n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    try:
        from xgboost import XGBClassifier

        zoo["xgboost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.9,
            colsample_bytree=0.9, random_state=42, n_jobs=-1, tree_method="hist",
            eval_metric="logloss",
        )
    except ImportError:
        pass
    return zoo


def build_pipeline(cfg, estimator, calibrate: bool = False):
    """Pipeline, optionally wrapped in probability calibration.

    Calibration matters here because the decision threshold is chosen on the cost curve.
    A miscalibrated score still ranks customers correctly (ROC-AUC is unaffected) but puts
    the cost minimum at the wrong place, so the chosen threshold would not transfer.
    """
    from sklearn.pipeline import Pipeline

    from .features import make_preprocessor

    pipe = Pipeline([("preprocessor", make_preprocessor(cfg)), ("classifier", estimator)])
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV

        return CalibratedClassifierCV(pipe, method="isotonic", cv=5)
    return pipe

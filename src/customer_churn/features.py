"""Feature assembly, schema coercion and the leakage guard.

The schema coercion in this module exists because of a real bug, worth stating plainly:

    The first version inferred the numeric/categorical split at fit time with
    `select_dtypes(include=["number", "bool"])`. Training data came from DuckDB, whose
    CSV sniffer types the 13 Yes/No columns as BOOLEAN -- so those columns went to
    StandardScaler. The Streamlit app built its input row from the raw CSV, where the
    same columns are the strings "Yes"/"No", and StandardScaler raised
    `ValueError: could not convert string to float: 'Yes'`.

    The model was not wrong, but its idea of the schema depended on which reader loaded
    the data. That is train/serve skew, and the pipeline could not have caught it,
    because at training time nothing was inconsistent.

The fix is two-part: the split is declared in `config.yaml` rather than inferred, and
every frame -- training or serving -- passes through `coerce_schema` first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


class LeakageError(ValueError):
    """Raised when an outcome-derived column reaches the model matrix."""


class SchemaError(ValueError):
    """Raised when a frame cannot be coerced to the declared schema."""


def assert_no_banned_features(columns: list[str], cfg: Config) -> None:
    banned = sorted(set(columns) & set(cfg.banned_features))
    if banned:
        raise LeakageError(
            f"{banned} is derived from the outcome (Churn_Category and Churn_Reason are "
            f"populated for 100% of churned and 0% of stayed customers); "
            f"refusing to build features."
        )


def coerce_schema(df: pd.DataFrame, cfg: Config, *, strict: bool = True) -> pd.DataFrame:
    """Cast a frame to the declared schema, whatever reader produced it.

    Idempotent by construction: a frame already in the target schema passes through
    unchanged, so it is safe to apply at both ends of the pipeline.

    - boolean columns are rendered back to their string labels ("Yes"/"No")
    - declared categoricals become `str`, with the SQL COALESCE defaults applied
    - declared numerics go through `pd.to_numeric`
    """
    out = df.copy()
    true_label, false_label = cfg.boolean_labels

    for col in cfg.categorical_features:
        if col not in out.columns:
            if strict:
                raise SchemaError(f"missing declared categorical column: {col}")
            continue

        series = out[col]
        if series.dtype == bool or str(series.dtype) in {"boolean", "bool"}:
            series = series.map({True: true_label, False: false_label})

        fill = cfg.null_fill.get(col)
        if fill is not None:
            series = series.fillna(fill)

        # Any remaining NaN would stringify to the literal "nan" and become a phantom
        # category, so it is filled with the same label the SQL would have used.
        if series.isna().any():
            if col in cfg.null_fill:
                series = series.fillna(cfg.null_fill[col])
            else:
                raise SchemaError(
                    f"{col} contains nulls but has no null_fill default in config.yaml"
                )
        out[col] = series.astype(str)

    for col in cfg.numeric_features:
        if col not in out.columns:
            if strict:
                raise SchemaError(f"missing declared numeric column: {col}")
            continue
        coerced = pd.to_numeric(out[col], errors="coerce")
        if coerced.isna().any() and not out[col].isna().any():
            bad = out.loc[coerced.isna(), col].unique()[:5]
            raise SchemaError(f"{col} declared numeric but holds non-numeric values: {bad}")
        out[col] = coerced.astype(float)

    return out


def schema_fingerprint(df: pd.DataFrame, cfg: Config) -> dict[str, str]:
    """The dtype of every model feature, for comparing a serving frame to training.

    Persisted alongside the model so a mismatch is detectable at load rather than at
    the first prediction.
    """
    return {c: str(df[c].dtype) for c in cfg.feature_names if c in df.columns}


def split_xy(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    y = (df[cfg.target] == cfg.positive_class).astype(int).to_numpy()
    X = df.drop(columns=[cfg.target], errors="ignore")
    X = X.drop(columns=[c for c in cfg.banned_features if c in X.columns], errors="ignore")
    assert_no_banned_features(list(X.columns), cfg)
    X = coerce_schema(X, cfg)
    return X[cfg.feature_names], y


def prepare_inference_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The serving-side counterpart of `split_xy`.

    Both the API and the dashboard route through this, so neither can invent its own
    interpretation of the schema.
    """
    X = df.drop(columns=[cfg.target], errors="ignore")
    X = X.drop(columns=[c for c in cfg.banned_features if c in X.columns], errors="ignore")
    X = coerce_schema(X, cfg)
    return X[cfg.feature_names]


def make_preprocessor(cfg: Config):
    """Column split comes from config, never from the dtypes of whatever was loaded.

    The original notebook used LabelEncoder fitted on the full dataset before splitting:
    the category vocabulary was learned from test rows, and an ordinal code implied an
    order that does not exist for State or Payment_Method.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        [
            ("num", StandardScaler(), cfg.numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cfg.categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

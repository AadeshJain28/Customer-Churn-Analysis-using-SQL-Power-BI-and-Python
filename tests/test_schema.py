"""Train/serve skew guard.

These exist because of a bug that got all the way to the dashboard: the numeric and
categorical split was inferred with `select_dtypes`, DuckDB typed the 13 Yes/No columns
as BOOLEAN, and the app fed the same columns in from the CSV as strings. StandardScaler
raised `could not convert string to float: 'Yes'`.

Nothing in the training pipeline could have caught it, because at training time nothing
was inconsistent. The only test that finds this class of bug is one that builds a frame
the way the *serving* path does and pushes it through the *training* schema.
"""

from __future__ import annotations

import pandas as pd
import pytest

from customer_churn.config import Config
from customer_churn.features import (
    SchemaError,
    coerce_schema,
    prepare_inference_frame,
    split_xy,
)

YES_NO_COLUMNS = [
    "Married", "Phone_Service", "Multiple_Lines", "Internet_Service",
    "Online_Security", "Online_Backup", "Device_Protection_Plan", "Premium_Support",
    "Streaming_TV", "Streaming_Movies", "Streaming_Music", "Unlimited_Data",
    "Paperless_Billing",
]


def test_the_yes_no_columns_are_declared_categorical(cfg):
    """The 13 columns DuckDB types as BOOLEAN must be categorical by declaration."""
    for col in YES_NO_COLUMNS:
        assert col in cfg.categorical_features, f"{col} is not declared categorical"
        assert col not in cfg.numeric_features


def test_schema_covers_every_modelling_column(modelling, cfg):
    expected = set(modelling.columns) - set(cfg.banned_features) - {cfg.target}
    assert set(cfg.feature_names) == expected, (
        f"undeclared: {sorted(expected - set(cfg.feature_names))}; "
        f"declared but absent: {sorted(set(cfg.feature_names) - expected)}"
    )


def test_boolean_and_string_frames_coerce_identically(modelling, cfg):
    """The heart of it: a DuckDB-style frame and a CSV-style frame must converge.

    Simulates DuckDB's BOOLEAN typing, then asserts both representations produce a
    byte-identical model matrix after coercion.
    """
    csv_style = modelling.copy()

    duck_style = modelling.copy()
    for col in YES_NO_COLUMNS:
        duck_style[col] = duck_style[col].map({"Yes": True, "No": False}).astype("boolean")

    from_csv, y_csv = split_xy(csv_style, cfg)
    from_duck, y_duck = split_xy(duck_style, cfg)

    assert list(from_csv.columns) == list(from_duck.columns)
    assert (y_csv == y_duck).all()
    pd.testing.assert_frame_equal(from_csv, from_duck, check_dtype=True)


def test_coercion_is_idempotent(modelling, cfg):
    once = coerce_schema(modelling.drop(columns=[cfg.target], errors="ignore"), cfg, strict=False)
    twice = coerce_schema(once, cfg, strict=False)
    pd.testing.assert_frame_equal(once, twice)


def test_declared_numerics_are_float_and_categoricals_are_str(modelling, cfg):
    X, _ = split_xy(modelling, cfg)
    for col in cfg.numeric_features:
        assert X[col].dtype.kind == "f", f"{col} is {X[col].dtype}, expected float"
    for col in cfg.categorical_features:
        assert X[col].dtype == object, f"{col} is {X[col].dtype}, expected object"
        assert X[col].map(type).eq(str).all(), f"{col} holds non-str values"


def test_no_phantom_nan_category_survives(modelling, cfg):
    """`astype(str)` on a NaN produces the string 'nan', a category the model never
    saw in training. The null_fill defaults must run first."""
    X, _ = split_xy(modelling, cfg)
    for col in cfg.categorical_features:
        assert "nan" not in set(X[col].unique()), f"{col} contains a phantom 'nan' category"


def test_null_fill_matches_the_sql_coalesce_defaults(cfg):
    """Drift guard between config.yaml and sql/04_production.sql.

    If the SQL fills Internet_Type with 'None' and Python fills it with 'No', a row
    loaded from the CSV becomes a different row from the same customer loaded from the
    view -- and only one of them matches what the model was trained on.
    """
    import re
    from pathlib import Path

    sql = (Path(__file__).resolve().parents[1] / "sql" / "04_production.sql").read_text()
    pairs = re.findall(r"COALESCE\(\s*(\w+)\s*,\s*'([^']*)'\s*\)\s+AS\s+(\w+)", sql)
    coalesce = {alias: default for _, default, alias in pairs}

    modelled = {k: v for k, v in coalesce.items() if k not in cfg.banned_features}
    assert modelled == cfg.null_fill, (
        f"SQL COALESCE defaults and config null_fill disagree.\n"
        f"  SQL:    {sorted(modelled.items())}\n"
        f"  config: {sorted(cfg.null_fill.items())}"
    )


def test_a_single_row_built_the_way_the_app_builds_it_survives(modelling, cfg):
    """End-to-end regression for the reported crash, without needing a trained model.

    The dashboard composes a row from `mode()` over the raw CSV. That row must reach
    the model matrix with the declared dtypes.
    """
    row = {c: modelling[c].mode()[0] for c in modelling.columns if c != cfg.target}
    frame = prepare_inference_frame(pd.DataFrame([row]), cfg)

    assert list(frame.columns) == cfg.feature_names
    assert len(frame) == 1
    for col in cfg.numeric_features:
        assert frame[col].dtype.kind == "f"
    for col in cfg.categorical_features:
        assert isinstance(frame[col].iloc[0], str)


def test_missing_column_is_reported_not_silently_dropped(modelling, cfg):
    with pytest.raises(SchemaError, match="Contract"):
        coerce_schema(modelling.drop(columns=["Contract"]), cfg)


def test_non_numeric_value_in_a_numeric_column_is_rejected(modelling, cfg):
    broken = modelling.copy()
    broken.loc[broken.index[0], "Monthly_Charge"] = "seventy"
    with pytest.raises(SchemaError, match="Monthly_Charge"):
        coerce_schema(broken, cfg)


def test_config_rejects_unquoted_yaml_booleans(cfg):
    """YAML 1.1 parses an unquoted `No` as False. Catch it at load, not at serve."""
    from customer_churn.config import Config as C

    broken = {**cfg.raw, "schema": {**cfg.raw["schema"]}}
    broken["schema"]["null_fill"] = {**cfg.null_fill, "Online_Backup": False}
    with pytest.raises(ValueError, match="quoted strings"):
        C(raw=broken).validate()

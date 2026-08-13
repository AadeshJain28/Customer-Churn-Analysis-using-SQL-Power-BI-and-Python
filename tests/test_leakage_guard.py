"""Churn_Category and Churn_Reason are populated only for churners."""

from __future__ import annotations

import pytest

from customer_churn.config import Config
from customer_churn.features import LeakageError, assert_no_banned_features, split_xy


def test_outcome_columns_perfectly_separate_the_classes(modelling, cfg):
    """The measurement that justifies banning them. If it stops being true, the ban
    should be revisited rather than silently kept."""
    for col in ["Churn_Category", "Churn_Reason"]:
        churned = modelling.loc[modelling[cfg.target] == "Churned", col].notna().mean()
        stayed = modelling.loc[modelling[cfg.target] == "Stayed", col].notna().mean()
        assert churned == 1.0 and stayed == 0.0, f"{col} no longer separates perfectly"


def test_split_xy_drops_banned_columns(modelling, cfg):
    X, y = split_xy(modelling, cfg)
    for col in cfg.banned_features:
        assert col not in X.columns
    assert cfg.target not in X.columns
    assert set(y.tolist()) <= {0, 1}


def test_assert_raises_on_banned_column(cfg):
    with pytest.raises(LeakageError, match="Churn_Reason"):
        assert_no_banned_features(["Age", "Churn_Reason"], cfg)


def test_config_rejects_empty_ban_list(cfg):
    broken = dict(cfg.raw)
    broken["banned_features"] = []
    with pytest.raises(ValueError):
        Config(raw=broken).validate()


def test_config_rejects_degenerate_costs(cfg):
    broken = dict(cfg.raw)
    broken["costs"] = {"intervention": 5.0, "missed_churner": 1.0}
    with pytest.raises(ValueError, match="missed_churner"):
        Config(raw=broken).validate()

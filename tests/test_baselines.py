"""Baselines are the load-bearing claim of this repo, so they get hand-checked."""

from __future__ import annotations

import numpy as np

from customer_churn.baselines import (
    all_baselines,
    flag_everyone,
    majority_class,
    month_to_month_rule,
)


def _y(df):
    return (df.Customer_Status == "Churned").astype(int).to_numpy()


def test_majority_class_matches_hand_derivation(toy):
    s = majority_class(_y(toy))
    assert s.accuracy == 0.6      # 6 of 10 stayed
    assert s.recall == 0.0        # flags nobody
    assert s.flagged_share == 0.0


def test_flag_everyone_has_perfect_recall_and_base_rate_precision(toy):
    s = flag_everyone(_y(toy))
    assert s.recall == 1.0
    assert s.precision == 0.4     # the base rate
    assert s.accuracy == 0.4


def test_month_to_month_rule_matches_hand_derivation(toy):
    s = month_to_month_rule(toy, _y(toy))
    assert s.recall == 0.75       # 3 of 4 churners
    assert s.precision == 0.6     # 3 of 5 flagged
    assert s.accuracy == 0.7      # 7 of 10 correct
    assert s.flagged_share == 0.5


def test_rule_beats_majority_on_recall_in_real_data(modelling):
    """The finding the README leads with. If a data refresh breaks it, fail here."""
    y = _y(modelling)
    rule = month_to_month_rule(modelling, y)
    assert rule.recall > 0.85, f"rule recall dropped to {rule.recall}"
    assert rule.flagged_share < 0.55


def test_all_baselines_returns_every_reference(modelling):
    out = all_baselines(modelling, _y(modelling))
    assert set(out.name) == {
        "majority_class", "flag_everyone", "rule_month_to_month", "rule_tenure_lt_6m"
    }
    assert (out.accuracy.between(0, 1)).all()

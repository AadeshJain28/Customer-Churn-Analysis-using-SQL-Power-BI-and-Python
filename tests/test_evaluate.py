from __future__ import annotations

import numpy as np

from customer_churn.evaluate import expected_cost


def test_expected_cost_hand_computed():
    # 4 customers: predict [1,1,0,0], truth [1,0,1,0].
    # Contacted = 2 -> 2 * 1.0 = 2.0. Missed churners = 1 -> 1 * 8.0 = 8.0. Total 10.0.
    y = np.array([1, 0, 1, 0])
    p = np.array([1, 1, 0, 0])
    assert expected_cost(y, p, intervention=1.0, missed=8.0) == 10.0


def test_contacting_nobody_costs_only_missed_churners():
    y = np.array([1, 1, 0, 0])
    p = np.zeros(4, dtype=int)
    assert expected_cost(y, p, 1.0, 8.0) == 16.0


def test_contacting_everyone_costs_only_interventions():
    y = np.array([1, 1, 0, 0])
    p = np.ones(4, dtype=int)
    assert expected_cost(y, p, 1.0, 8.0) == 4.0


def test_cost_is_monotone_in_missed_churner_price():
    y = np.array([1, 1, 0, 0])
    p = np.array([1, 0, 0, 0])
    cheap = expected_cost(y, p, 1.0, 2.0)
    dear = expected_cost(y, p, 1.0, 20.0)
    assert dear > cheap, "raising the cost of a miss must raise total cost"

"""Tests for rq1_itemset.mine rule filtering."""
from __future__ import annotations

import pandas as pd

from rq1_itemset.mine import mine_rules


def test_mine_rules_keeps_env_to_disease_only():
    # Hand-crafted panel: env item pm_Q4 is perfectly correlated with a
    # disease flag, temperature Q1 is noise, and one disease item is also
    # correlated with another disease item (must be filtered out — rules
    # whose antecedent contains a disease item should not survive).
    n = 20
    rows = []
    for i in range(n):
        hit = i < 10
        rows.append(
            {
                "pm_Q1": not hit,
                "pm_Q4": hit,
                "temp_Q1": i % 2 == 0,
                "temp_Q4": i % 2 == 1,
                "J00-J99_high": hit,
                "I00-I99_high": hit,
            }
        )
    trx = pd.DataFrame(rows)
    env = ["pm_Q1", "pm_Q4", "temp_Q1", "temp_Q4"]
    disease = ["J00-J99_high", "I00-I99_high"]
    rules = mine_rules(trx, env, disease, min_support=0.2, min_confidence=0.5)

    assert not rules.empty
    # Every antecedent is a subset of env items
    for ants in rules["antecedents"]:
        assert set(ants).issubset(set(env))
    # Every consequent contains at least one disease item
    for cons in rules["consequents"]:
        assert set(cons) & set(disease)
    # The strong rule pm_Q4 -> J00-J99_high must appear with confidence 1.0
    strong = rules[
        rules["antecedents"].map(lambda s: s == frozenset({"pm_Q4"}))
        & rules["consequents"].map(lambda s: "J00-J99_high" in s)
    ]
    assert len(strong) >= 1
    assert strong.iloc[0]["confidence"] == 1.0


def test_mine_rules_returns_empty_when_nothing_clears_thresholds():
    # Independent columns, no rule should clear confidence 0.99.
    import numpy as np

    rng = np.random.default_rng(0)
    n = 50
    trx = pd.DataFrame(
        {
            "pm_Q4": rng.random(n) > 0.5,
            "J00-J99_high": rng.random(n) > 0.5,
        }
    )
    rules = mine_rules(
        trx, ["pm_Q4"], ["J00-J99_high"], min_support=0.1, min_confidence=0.99
    )
    assert rules.empty
    # Column schema preserved even on empty output
    assert list(rules.columns) == [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
        "antecedent_support",
        "consequent_support",
    ]

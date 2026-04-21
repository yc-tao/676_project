"""Tests for rq1_itemset.bridge parsing and join."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rq1_itemset.bridge import cross_link, extract_env_chapter


def test_extract_env_chapter_simple():
    env, chap = extract_env_chapter("pm25_Q4", "J00-J99_high")
    assert env == "pm25"
    assert chap == "J00-J99"


def test_extract_env_chapter_multi_item_antecedent_takes_first_alpha():
    env, chap = extract_env_chapter("temp_Q1, aerosol_Q4", "I00-I99_high, temp_Q1")
    # sorted alphabetically -> "aerosol_Q4" first; consequent env token ignored
    assert env == "aerosol"
    assert chap == "I00-I99"


def test_extract_env_chapter_no_disease_in_consequent():
    env, chap = extract_env_chapter("pm25_Q4", "aerosol_Q1")
    assert env == "pm25"
    assert chap is None


def test_cross_link_joins_dlnm(tmp_path: Path):
    rules = pd.DataFrame(
        {
            "antecedents": ["pm25_Q4", "temp_Q1"],
            "consequents": ["J00-J99_high", "E00-E89_high"],
            "support": [0.1, 0.1],
            "confidence": [0.8, 0.6],
            "lift": [2.5, 1.8],
        }
    )
    dlnm = pd.DataFrame(
        {
            "exposure": ["pm25", "temp", "foo"],
            "chapter": ["J00-J99", "E00-E89", "I00-I99"],
            "log_rr": [1.2, -0.9, 0.3],
            "q": [0.01, 0.05, 0.8],
            "converged": [True, True, True],
        }
    )
    rules_path = tmp_path / "rules.csv"
    dlnm_path = tmp_path / "dlnm.csv"
    rules.to_csv(rules_path, index=False)
    dlnm.to_csv(dlnm_path, index=False)

    out = cross_link(rules_path, dlnm_path)
    assert len(out) == 2
    assert set(out["exposure"]) == {"pm25", "temp"}
    pm25_row = out[out["exposure"] == "pm25"].iloc[0]
    assert pm25_row["dlnm_log_rr"] == 1.2
    # pm25 pair has the largest abs(log_rr) -> DLNM rank 1
    assert int(pm25_row["dlnm_rank_abs"]) == 1

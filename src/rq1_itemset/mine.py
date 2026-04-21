"""FP-Growth sweep over the CBSA-year transaction panel.

Generates frequent itemsets via FP-Growth, converts them to association
rules, and keeps only rules whose antecedent is purely environmental
(``<col>_Q{1..4}`` items) and whose consequent contains at least one
disease item (``<chapter>_high``).

Thresholds are chosen for a thin 92-row panel: min support 3/92 ≈ 0.033,
min confidence 0.5. These are the lowest values that still make the
"support floor ≥ 3 transactions" claim from the checkpoint notebook.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth


MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.5
MAX_LEN = 3


def mine_rules(
    transactions: pd.DataFrame,
    env_items: Sequence[str],
    disease_items: Sequence[str],
    *,
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    max_len: int = MAX_LEN,
) -> pd.DataFrame:
    """Run FP-Growth and return rules with env antecedent, disease consequent.

    Returned columns: antecedents, consequents, support, confidence, lift,
    antecedent_support, consequent_support. Sorted by lift desc then
    confidence desc. Empty DataFrame if no rules clear thresholds.

    `max_len` caps itemset size. With 112 items on 92 rows, uncapped
    enumeration is combinatorial; 3 keeps the search tractable while still
    letting a 2-item env antecedent imply a disease flag.
    """
    item_cols = list(env_items) + list(disease_items)
    tx = transactions[item_cols].astype(bool)

    freq = fpgrowth(tx, min_support=min_support, use_colnames=True, max_len=max_len)
    if freq.empty:
        return _empty_rules()

    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return _empty_rules()

    env_set = set(env_items)
    disease_set = set(disease_items)

    def keep(row) -> bool:
        ants = set(row["antecedents"])
        cons = set(row["consequents"])
        return ants.issubset(env_set) and bool(cons & disease_set)

    filt = rules[rules.apply(keep, axis=1)].copy()
    if filt.empty:
        return _empty_rules()

    filt = filt[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
            "antecedent support",
            "consequent support",
        ]
    ].rename(
        columns={
            "antecedent support": "antecedent_support",
            "consequent support": "consequent_support",
        }
    )
    return filt.sort_values(
        ["lift", "confidence", "support"], ascending=[False, False, False]
    ).reset_index(drop=True)


def _empty_rules() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
            "antecedent_support",
            "consequent_support",
        ]
    )


@dataclass
class SweepResult:
    transactions: pd.DataFrame
    env_items: list[str]
    disease_items: list[str]
    rules: pd.DataFrame


def run(
    dataset_root: Path | str = "sathealth_dataset",
    outcomes_path: Path | str = "sathealth_dataset/icdl1_prev_ohio.csv",
    cbsa_list: Iterable[int] = (
        10420, 15940, 17140, 17460, 18140, 19380, 26580,
        30620, 31900, 44220, 45780, 48260, 48540, 49660,
    ),
    chapters: Iterable[str] = ("J00-J99", "I00-I99", "F01-F99", "E00-E89"),
    results_dir: Path | str = "results",
) -> SweepResult:
    """End-to-end: build panel, mine rules, write CSV to `results/`."""
    from rq1_itemset.data import build_transactions

    trx, env_cols, disease_cols = build_transactions(
        dataset_root,
        cbsa_list=cbsa_list,
        chapters=chapters,
        outcomes_path=outcomes_path,
    )
    rules = mine_rules(trx, env_cols, disease_cols)

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    rules_out = rules.copy()
    rules_out["antecedents"] = rules_out["antecedents"].map(
        lambda s: ", ".join(sorted(s))
    )
    rules_out["consequents"] = rules_out["consequents"].map(
        lambda s: ", ".join(sorted(s))
    )
    rules_out.to_csv(out / "itemset_rules.csv", index=False)

    print(f"transactions: {trx.shape}")
    print(f"rules kept: {len(rules)}")
    if len(rules) > 0:
        print(rules_out.head(15).to_string(index=False))
    return SweepResult(transactions=trx, env_items=env_cols, disease_items=disease_cols, rules=rules)

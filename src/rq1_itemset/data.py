"""Build the CBSA-year transaction panel for FP-Growth.

Each row is one CBSA-year; each column is a binary item. Environmental
variables (monthly) are aggregated to yearly means and discretized into
global quartile bins. Disease prevalences (yearly, ICD L1) are binarized
against a per-chapter high-prevalence threshold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from rq1_dlnm.data import load_env_monthly, prune_exposures

KEY = ["CBSAFP", "year"]


def aggregate_env_yearly(env_monthly: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Collapse monthly env to CBSA-year means on `cols`."""
    return (
        env_monthly.groupby(KEY, as_index=False)[list(cols)]
        .mean(numeric_only=True)
        .sort_values(KEY)
        .reset_index(drop=True)
    )


def discretize_quartiles(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Return a one-hot DataFrame with items `<col>_Q{1..4}` for each col.

    Uses global quantile bins across the whole panel (not per-CBSA). Ties and
    constant columns are handled by `pd.qcut(..., duplicates='drop')`, which
    may produce fewer than 4 bins; items are named by the quartile rank
    actually observed (Q1…Qk).
    """
    pieces = []
    for c in cols:
        try:
            cats = pd.qcut(df[c], q=4, labels=False, duplicates="drop")
        except ValueError:
            continue
        if cats.isna().all():
            continue
        ranks = cats.astype("Int64") + 1
        dummies = pd.get_dummies(ranks.astype("object"), prefix=c, prefix_sep="_Q")
        dummies.columns = [str(col).replace("_Q<NA>", "_Qmiss") for col in dummies.columns]
        dummies = dummies.drop(columns=[x for x in dummies.columns if x.endswith("_Qmiss")], errors="ignore")
        pieces.append(dummies.astype(bool))
    if not pieces:
        return pd.DataFrame(index=df.index)
    return pd.concat(pieces, axis=1)


def binarize_outcomes(
    outcomes: pd.DataFrame,
    chapters: Iterable[str],
    *,
    quantile: float = 2 / 3,
) -> pd.DataFrame:
    """Return a wide CBSA-year table with boolean `<chapter>_high` items.

    A (CBSA, year, chapter) row is flagged high if its prevalence is at or
    above the `quantile`-th quantile of prevalence within that chapter across
    the whole panel. Default 2/3 (top tertile) gives a roughly 1-in-3 prior.
    """
    df = outcomes[outcomes["code"].isin(list(chapters))].copy()
    thresholds = df.groupby("code")["prevalence"].quantile(quantile)
    df["high"] = df["prevalence"] >= df["code"].map(thresholds)
    wide = df.pivot_table(
        index=KEY, columns="code", values="high", aggfunc="first"
    ).fillna(False)
    wide.columns = [f"{c}_high" for c in wide.columns]
    return wide.reset_index().astype({c: bool for c in wide.columns})


def build_transactions(
    dataset_root: Path | str,
    *,
    cbsa_list: Iterable[int],
    chapters: Iterable[str],
    outcomes_path: Path | str,
    env_candidate_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """End-to-end builder: return (transactions, env_items, disease_items).

    `transactions` is a boolean DataFrame indexed by CBSA-year plus one
    column per item. `env_items` and `disease_items` are the column names of
    the two item groups so downstream code can filter rules by antecedent
    type.
    """
    env = load_env_monthly(dataset_root, cbsa_list)
    if env_candidate_cols is None:
        env_candidate_cols = [
            c for c in env.columns
            if c not in {"CBSAFP", "year", "month"} and env[c].dtype != "object"
        ]
    kept = prune_exposures(env, list(env_candidate_cols))
    env_year = aggregate_env_yearly(env, kept)
    env_items = discretize_quartiles(env_year[kept], kept)
    env_items.insert(0, "year", env_year["year"].values)
    env_items.insert(0, "CBSAFP", env_year["CBSAFP"].values)

    outcomes = pd.read_csv(
        outcomes_path, usecols=["CBSAFP", "year", "code", "prevalence"]
    )
    outcomes = outcomes[outcomes["code"].isin(list(chapters))]
    outcomes = outcomes[outcomes["CBSAFP"].isin(list(cbsa_list))]
    disease_items = binarize_outcomes(outcomes, chapters)

    trx = env_items.merge(disease_items, on=KEY, how="inner").sort_values(KEY).reset_index(drop=True)
    env_cols = [c for c in trx.columns if c not in KEY and not c.endswith("_high")]
    disease_cols = [c for c in trx.columns if c.endswith("_high")]
    return trx, env_cols, disease_cols

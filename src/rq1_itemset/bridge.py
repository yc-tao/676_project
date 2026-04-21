"""Cross-link FP-Growth rules with DLNM cumulative-RR rankings.

The two pipelines ask different questions — DLNM wants a dose-response
curve per (exposure, chapter) pair, FP-Growth wants co-occurrence of
discrete regime tokens. This module builds a small table that lets the
narrative connect them: for each FP-Growth rule, look up the DLNM
cumulative log-RR for the underlying (exposure, chapter) pair so a
reviewer can see that the top itemset rules ride the same signals the
DLNM surfaces.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ENV_ITEM_RE = re.compile(r"^(?P<col>.+)_Q(?P<q>[1-4])$")
DISEASE_ITEM_RE = re.compile(r"^(?P<chapter>[A-Z]\d{2}-[A-Z]?\d{2}[A-Z]?)_high$")


def parse_rule_tokens(items_str: str) -> list[str]:
    """Split a comma-separated rule token string into a list of items."""
    return [tok.strip() for tok in items_str.split(",") if tok.strip()]


def extract_env_chapter(ants_str: str, cons_str: str) -> tuple[str | None, str | None]:
    """Pull the first env column + first disease chapter from a rule.

    If the rule's antecedent has multiple env items, we take the first
    alphabetically (deterministic). Ditto for consequent disease items.
    Returns (None, None) if the expected structure is not present.
    """
    env_col = None
    for tok in sorted(parse_rule_tokens(ants_str)):
        m = ENV_ITEM_RE.match(tok)
        if m:
            env_col = m.group("col")
            break
    chapter = None
    for tok in sorted(parse_rule_tokens(cons_str)):
        m = DISEASE_ITEM_RE.match(tok)
        if m:
            chapter = m.group("chapter")
            break
    return env_col, chapter


def cross_link(
    rules_path: Path | str,
    dlnm_summary_path: Path | str,
) -> pd.DataFrame:
    """Join FP-Growth rules to DLNM (exposure, chapter) rankings.

    Returned columns: antecedents, consequents, support, confidence, lift,
    exposure, chapter, dlnm_log_rr, dlnm_q, dlnm_rank_abs.

    `dlnm_rank_abs` is the rank (1=strongest) of abs(log_rr) across all
    108 DLNM pairs, so the bridge table can highlight rules whose
    underlying pair is also near the top of the DLNM ranking.
    """
    rules = pd.read_csv(rules_path)
    dlnm = pd.read_csv(dlnm_summary_path)
    dlnm["abs_log_rr"] = dlnm["log_rr"].abs()
    dlnm = dlnm.sort_values("abs_log_rr", ascending=False).reset_index(drop=True)
    dlnm["rank_abs"] = dlnm.index + 1

    parsed = rules.apply(
        lambda r: pd.Series(extract_env_chapter(r["antecedents"], r["consequents"])),
        axis=1,
    )
    parsed.columns = ["exposure", "chapter"]
    joined = pd.concat([rules, parsed], axis=1)

    dlnm_lookup = dlnm.set_index(["exposure", "chapter"])[["log_rr", "q", "rank_abs"]]
    dlnm_lookup.columns = ["dlnm_log_rr", "dlnm_q", "dlnm_rank_abs"]
    joined = joined.merge(
        dlnm_lookup, how="left", left_on=["exposure", "chapter"], right_index=True
    )

    return joined[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
            "exposure",
            "chapter",
            "dlnm_log_rr",
            "dlnm_q",
            "dlnm_rank_abs",
        ]
    ]


def run(
    rules_path: Path | str = "results/itemset_rules.csv",
    dlnm_summary_path: Path | str = "results/summary_dlnm.csv",
    out_path: Path | str = "results/bridge.csv",
    top_n: int = 20,
) -> pd.DataFrame:
    """Write the cross-linked bridge table and return the top-N rows by lift."""
    bridge = cross_link(rules_path, dlnm_summary_path)
    bridge.to_csv(out_path, index=False)

    top = bridge.sort_values("lift", ascending=False).head(top_n)
    print(f"rules linked: {len(bridge)}")
    print(f"rules with a matched DLNM pair: {bridge['dlnm_log_rr'].notna().sum()}")
    print(f"\n--- top {top_n} rules by lift, with DLNM cross-reference ---")
    for _, r in top.iterrows():
        marker = (
            f"DLNM rank #{int(r['dlnm_rank_abs'])} "
            f"(log_rr={r['dlnm_log_rr']:+.2f}, q={r['dlnm_q']:.3g})"
            if pd.notna(r["dlnm_log_rr"])
            else "no matching DLNM pair"
        )
        print(
            f"  {r['antecedents']}  =>  {r['consequents']}"
            f"   lift={r['lift']:.2f} conf={r['confidence']:.2f}   {marker}"
        )
    return top

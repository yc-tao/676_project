"""
EDA script for the SatHealth Dataset.
Produces a Markdown report (eda_report.md) with summary statistics,
missing-value analysis, distributions, correlations, and temporal trends.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)
DATASET_DIR = Path("sathealth_dataset")
GEO_LEVELS = ["CBSA", "County", "CT", "ZCTA"]
ENV_FILES = ["airquality.csv", "climate.csv", "greenery.csv", "landcover.csv"]

report_lines = []

def md(text=""):
    report_lines.append(text)

def md_table(df, max_rows=30):
    """Append a DataFrame as a markdown table."""
    df = df.head(max_rows)
    md(df.to_markdown(index=True))
    md()

def save_fig(name):
    path = OUTPUT_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

# ── 1. Column Dictionary ─────────────────────────────────────────────────────
md("# SatHealth Dataset — Exploratory Data Analysis")
md()
md("## 1. Column Dictionary Overview")
md()
col_dict = pd.read_csv(DATASET_DIR / "column_dictionary.csv")
md(f"Total entries: **{len(col_dict)}**")
md()
md("### Columns per table")
md()
md_table(col_dict.groupby("table").size().rename("count").to_frame())
md()
md("### Data types breakdown")
md()
md_table(col_dict.groupby("dtype").size().rename("count").to_frame())

# ── 2. Environmental data (County level as representative) ────────────────────
md()
md("## 2. Environmental Data (County Level)")
md()

env_dfs = {}
for fname in ENV_FILES:
    fpath = DATASET_DIR / "County" / fname
    if fpath.exists():
        env_dfs[fname.replace(".csv", "")] = pd.read_csv(fpath)

for name, df in env_dfs.items():
    md(f"### 2.{list(env_dfs).index(name)+1} {name.title()}")
    md()
    md(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    md()

    # Basic stats for numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude ID / year / month from stats
    skip = {"COUNTYFP", "CBSAFP", "CTFP", "ZCTA5", "year", "month"}
    stat_cols = [c for c in num_cols if c not in skip]

    if stat_cols:
        md("#### Summary Statistics")
        md()
        md_table(df[stat_cols].describe().round(4).T)

        # Missing values
        missing = df[stat_cols].isnull().sum()
        if missing.sum() > 0:
            md("#### Missing Values")
            md()
            md_table(missing[missing > 0].rename("missing_count").to_frame())

        # Distribution plot
        n = len(stat_cols)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for i, col in enumerate(stat_cols):
            ax = axes[i]
            df[col].dropna().hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
            ax.set_title(col, fontsize=8)
            ax.tick_params(labelsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"{name.title()} — Distributions (County)", fontsize=11)
        figpath = save_fig(f"dist_{name}")
        md(f"![{name} distributions]({figpath})")
        md()

# ── 3. Correlation heatmap for climate variables ─────────────────────────────
md("## 3. Correlation Analysis — Climate Variables (County)")
md()
climate_df = env_dfs.get("climate")
if climate_df is not None:
    skip = {"COUNTYFP", "CBSAFP", "CTFP", "ZCTA5", "year", "month"}
    clim_cols = [c for c in climate_df.select_dtypes(include=[np.number]).columns if c not in skip]
    corr = climate_df[clim_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, ax=ax,
                cbar_kws={"shrink": 0.6},
                xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=6)
    ax.set_title("Climate Variable Correlations (County Level)", fontsize=11)
    figpath = save_fig("corr_climate")
    md(f"![Climate correlation heatmap]({figpath})")
    md()

    # Top positive and negative correlations
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().reset_index()
    pairs.columns = ["var1", "var2", "correlation"]
    pairs["abs_corr"] = pairs["correlation"].abs()
    md("### Top 10 strongest correlations")
    md()
    md_table(pairs.nlargest(10, "abs_corr")[["var1", "var2", "correlation"]].reset_index(drop=True))

# ── 4. Temporal trends ────────────────────────────────────────────────────────
md("## 4. Temporal Trends (County Level, Annual Means)")
md()

trend_vars = {
    "airquality": ["particulate_matter_d_less_than_25_um_surface", "NO2_column_number_density"],
    "climate": ["temperature_2m", "total_precipitation_sum"],
    "greenery": ["NDVI", "NDVI_binary"],
}

for name, cols in trend_vars.items():
    df = env_dfs.get(name)
    if df is None:
        continue
    available = [c for c in cols if c in df.columns]
    if not available:
        continue
    annual = df.groupby("year")[available].mean()
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 3.5))
    if len(available) == 1:
        axes = [axes]
    for ax, col in zip(axes, available):
        annual[col].plot(ax=ax, marker="o", color="teal")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{name.title()} — Annual Trends", fontsize=11)
    figpath = save_fig(f"trend_{name}")
    md(f"![{name} trends]({figpath})")
    md()

# ── 5. Landcover composition ─────────────────────────────────────────────────
md("## 5. Landcover Composition (County Level)")
md()
lc = env_dfs.get("landcover")
if lc is not None:
    cover_cols = [c for c in lc.columns if "coverfraction" in c]
    means = lc[cover_cols].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    means.plot.bar(ax=ax, color="forestgreen", edgecolor="white")
    ax.set_ylabel("Mean Cover Fraction (%)")
    ax.set_title("Average Landcover Composition — Ohio Counties")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    figpath = save_fig("landcover_bar")
    md(f"![Landcover composition]({figpath})")
    md()
    md("### Landcover Summary")
    md()
    md_table(means.rename("mean_pct").to_frame().round(2))

# ── 6. Social Deprivation Index (SDI) ────────────────────────────────────────
md("## 6. Social Deprivation Index (County Level)")
md()
sdi_path = DATASET_DIR / "County" / "sdi.csv"
if sdi_path.exists():
    sdi = pd.read_csv(sdi_path)
    md(f"Shape: **{sdi.shape[0]} rows × {sdi.shape[1]} columns**")
    md()

    score_cols = [c for c in sdi.columns if c.endswith("_score") or c == "sdi"]
    pct_cols = [c for c in sdi.columns if c.startswith("pct")]

    md("### Score columns (1-100 centile)")
    md()
    md_table(sdi[score_cols].describe().round(2).T)

    fig, ax = plt.subplots(figsize=(8, 4))
    sdi[score_cols].boxplot(ax=ax, vert=True, patch_artist=True,
                            boxprops=dict(facecolor="lightskyblue"))
    ax.set_title("SDI Score Distributions (County)")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    figpath = save_fig("sdi_boxplot")
    md(f"![SDI boxplot]({figpath})")
    md()

    # SDI correlation with percentage columns
    if pct_cols:
        md("### Percentage columns")
        md()
        md_table(sdi[pct_cols].describe().round(4).T)

# ── 7. Disease prevalence (ICD Level 1) ──────────────────────────────────────
md("## 7. Disease Prevalence — ICD-10 Level 1 (CBSA)")
md()
icd1_path = DATASET_DIR / "icdl1_prev_ohio.csv"
if icd1_path.exists():
    icd1 = pd.read_csv(icd1_path)
    md(f"Shape: **{icd1.shape[0]} rows × {icd1.shape[1]} columns**")
    md()

    # Average prevalence by code
    avg_prev = icd1.groupby("code")["prevalence"].mean().sort_values(ascending=False)
    md("### Average Prevalence by ICD-10 L1 Code")
    md()
    md_table(avg_prev.rename("mean_prevalence").to_frame().round(4))

    # Top 10 bar chart
    top10 = avg_prev.head(10)
    fig, ax = plt.subplots(figsize=(9, 4))
    top10.plot.barh(ax=ax, color="coral", edgecolor="white")
    ax.set_xlabel("Mean Prevalence")
    ax.set_title("Top 10 ICD-10 L1 Disease Categories by Prevalence")
    ax.invert_yaxis()
    figpath = save_fig("icd1_top10")
    md(f"![ICD-10 L1 top 10]({figpath})")
    md()

    # Temporal trend for top 5 codes
    top5_codes = avg_prev.head(5).index.tolist()
    sub = icd1[icd1["code"].isin(top5_codes)]
    pivot = sub.pivot_table(index="year", columns="code", values="prevalence", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(ax=ax, marker="o")
    ax.set_title("Prevalence Trends — Top 5 ICD-10 L1 Codes")
    ax.set_ylabel("Mean Prevalence")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    figpath = save_fig("icd1_trends")
    md(f"![ICD-10 L1 trends]({figpath})")
    md()

# ── 8. Geographic scale comparison ───────────────────────────────────────────
md("## 8. Geographic Scale Comparison")
md()
md("Number of unique geographic units per level:")
md()
scale_info = []
for level in GEO_LEVELS:
    aq_path = DATASET_DIR / level / "airquality.csv"
    if aq_path.exists():
        tmp = pd.read_csv(aq_path, nrows=0)
        geo_col = [c for c in tmp.columns if c.endswith("FP") or c.startswith("ZCTA")][0]
        tmp_full = pd.read_csv(aq_path, usecols=[geo_col])
        n_units = tmp_full[geo_col].nunique()
        scale_info.append({"Level": level, "GeoID Column": geo_col, "Unique Units": n_units})
scale_df = pd.DataFrame(scale_info)
md_table(scale_df.set_index("Level"))

# ── 9. Missing value heatmap across all County files ─────────────────────────
md("## 9. Missing Value Summary (County Level)")
md()
missing_summary = []
for name, df in env_dfs.items():
    total = len(df)
    for col in df.columns:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            missing_summary.append({
                "file": name,
                "column": col,
                "missing_count": n_miss,
                "missing_pct": round(100 * n_miss / total, 2),
            })
if missing_summary:
    miss_df = pd.DataFrame(missing_summary)
    md_table(miss_df.set_index(["file", "column"]))
else:
    md("No missing values found in County-level environmental files.")
md()

# ── 10. Google Map Points overview ───────────────────────────────────────────
md("## 10. Google Map Points Overview")
md()
gmp_path = DATASET_DIR / "google_map_points.csv"
if gmp_path.exists():
    gmp = pd.read_csv(gmp_path)
    md(f"Shape: **{gmp.shape[0]} rows × {gmp.shape[1]} columns**")
    md()
    md(f"- Unique counties: **{gmp['COUNTYFP'].nunique()}**")
    md(f"- Longitude range: **{gmp['lon'].min():.4f}** to **{gmp['lon'].max():.4f}**")
    md(f"- Latitude range: **{gmp['lat'].min():.4f}** to **{gmp['lat'].max():.4f}**")
    md()

    fig, ax = plt.subplots(figsize=(8, 6))
    sample = gmp.sample(n=min(10000, len(gmp)), random_state=42)
    ax.scatter(sample["lon"], sample["lat"], s=0.5, alpha=0.4, c="navy")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Google Map Grid Points (10k sample)")
    ax.set_aspect("equal")
    figpath = save_fig("gmap_points")
    md(f"![Google Map points]({figpath})")
    md()

# ── Write report ──────────────────────────────────────────────────────────────
report_path = Path("eda_report.md")
report_path.write_text("\n".join(report_lines))
print(f"EDA report written to {report_path}")
print(f"Figures saved to {OUTPUT_DIR}/")

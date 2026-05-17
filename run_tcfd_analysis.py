"""
TCFD Descriptive Analysis
--------------------------
Reads all TCFD classification output files and produces a standalone
descriptive analysis of TCFD pillar distributions across 40 European banks
for 2023 and 2024.

Analyses:
  1. Per-bank-year pillar distribution (governance / strategy / risk / metrics)
  2. Sample-level averages and paired t-tests per pillar (2023 vs 2024)
  3. Pillar balance (normalised Shannon entropy)
  4. Correlation of pillar shares with DQI components
  5. Year-over-year delta in metrics/targets share per bank

Inputs:
    results/tcfd/          — TCFD output .xlsx files (from climatebert_tcfd_colab.ipynb)
    results/dqi/<latest>   — DQI file (output of build_dqi.py)

Output:
    results/tcfd/tcfd_analysis_<date>.xlsx   — four sheets:
        PillarSummary     : per-bank-year pillar shares + balance
        SampleAverages    : mean per pillar by year + paired t-tests
        Correlations      : Pearson r between pillar shares and DQI components
        Deltas            : per-bank 2023→2024 change per pillar

Usage:
    python run_tcfd_analysis.py
    python run_tcfd_analysis.py --tcfd_dir results/tcfd --dqi results/dqi/dqi_2026-05-02.xlsx
"""

import os
import re
import glob
import argparse
import warnings
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

TCFD_DIR  = os.path.join("results", "tcfd")
DQI_DIR   = os.path.join("results", "dqi")
OUT_DIR   = os.path.join("results", "tcfd")

PILLARS           = ["governance", "strategy", "risk", "metrics"]
DETECTOR_POSITIVE = "yes"


# ── Key normalisation ──────────────────────────────────────────────────────────

def _norm(key) -> str:
    """
    Normalise a bank key for merging: lowercase, strip whitespace, and collapse
    hyphens and spaces to nothing.

    TCFD output files derive bank names from the Colab notebook and store them
    with spaces (e.g. 'abn amro'), whereas the DQI file uses PDF-filename-derived
    keys with hyphens (e.g. 'abn-amro').  Stripping both separators makes the
    merge key format-agnostic: 'abn amro' and 'abn-amro' both become 'abnamro'.
    """
    return re.sub(r'[\s\-]+', '', str(key)).strip().lower()


# ── Step 1: Load all TCFD files ────────────────────────────────────────────────

def load_tcfd_files(tcfd_dir: str) -> pd.DataFrame:
    """
    Read all TCFD output Excel files in tcfd_dir.
    Returns a summary DataFrame with one row per bank-year.
    """
    files = sorted([
        f for f in os.listdir(tcfd_dir)
        if f.endswith(".xlsx") and "_tcfd_" in f and "tcfd_analysis" not in f
    ])

    if not files:
        raise FileNotFoundError(
            f"No TCFD output files found in {tcfd_dir}. "
            "Run climatebert_tcfd_colab.ipynb first."
        )

    print(f"  Found {len(files)} TCFD file(s) in {tcfd_dir}")

    rows = []
    for fname in files:
        path = os.path.join(tcfd_dir, fname)
        df   = pd.read_excel(path)

        if "detector_label" not in df.columns or "tcfd_label" not in df.columns:
            print(f"  ⚠  Skipping {fname} — missing required columns")
            continue

        bank = _norm(df["bank"].iloc[0])
        year = int(df["year"].iloc[0])

        climate   = df[df["detector_label"] == DETECTOR_POSITIVE]
        n_total   = len(df)
        n_climate = len(climate)

        if n_climate == 0:
            print(f"  ⚠  {fname}: no climate paragraphs")
            continue

        counts = climate["tcfd_label"].value_counts()
        props  = {p: counts.get(p, 0) / n_climate for p in PILLARS}

        # Pillar balance: normalised Shannon entropy (0 = one pillar, 1 = perfectly even)
        probs_arr = np.array([props[p] for p in PILLARS])
        probs_nz  = probs_arr[probs_arr > 0]
        entropy   = -np.sum(probs_nz * np.log(probs_nz))
        balance   = round(entropy / np.log(4), 4)

        rows.append({
            "bank":            bank,
            "year":            year,
            "n_total":         n_total,
            "n_climate":       n_climate,
            "coverage":        round(n_climate / n_total, 4),
            "pct_governance":  round(props["governance"], 4),
            "pct_strategy":    round(props["strategy"], 4),
            "pct_risk":        round(props["risk"], 4),
            "pct_metrics":     round(props["metrics"], 4),
            "balance":         balance,
            "dominant_pillar": max(props, key=props.get),
        })

    df_out = pd.DataFrame(rows).sort_values(["bank", "year"]).reset_index(drop=True)
    print(f"  ✓ Loaded {len(df_out)} bank-year observations "
          f"({df_out['bank'].nunique()} banks)")
    return df_out


# ── Step 2: Sample averages + paired t-tests ──────────────────────────────────

def compute_sample_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sample-wide means per year and paired t-tests (2024 vs 2023).
    """
    d23 = df[df["year"] == 2023].set_index("bank")
    d24 = df[df["year"] == 2024].set_index("bank")
    common = d23.index.intersection(d24.index)

    rows = []
    for col_base, label in [
        ("pct_governance", "Governance"),
        ("pct_strategy",   "Strategy"),
        ("pct_risk",       "Risk management"),
        ("pct_metrics",    "Metrics / targets"),
        ("balance",        "Pillar balance (entropy)"),
    ]:
        delta  = d24.loc[common, col_base] - d23.loc[common, col_base]
        t_stat, p_val = stats.ttest_rel(
            d24.loc[common, col_base], d23.loc[common, col_base]
        )
        rows.append({
            "Metric":    label,
            "N pairs":   len(common),
            "Mean 2023": round(d23.loc[common, col_base].mean(), 4),
            "Mean 2024": round(d24.loc[common, col_base].mean(), 4),
            "Delta":     round(delta.mean(), 4),
            "Std delta": round(delta.std(), 4),
            "t-stat":    round(t_stat, 4),
            "p-value":   round(p_val, 4),
            "Stars": (
                "***" if p_val < 0.01 else
                "**"  if p_val < 0.05 else
                "*"   if p_val < 0.10 else ""
            ),
        })
    return pd.DataFrame(rows)


# ── Step 3: Correlations with DQI ─────────────────────────────────────────────

def compute_correlations(df_tcfd: pd.DataFrame, dqi_path: str) -> pd.DataFrame:
    """
    Merge TCFD summary with DQI file and compute Pearson correlations.

    Both sides are normalised to lowercase before merging, so the join works
    regardless of how casing differs between the two files.
    """
    dqi = pd.read_excel(dqi_path)

    # Normalise keys on both sides
    df_tcfd = df_tcfd.copy()
    df_tcfd["bank_key"] = df_tcfd["bank"].map(_norm)

    dqi["bank_key"] = dqi["bank"].map(_norm)
    dqi["year"]     = dqi["year"].astype(int)

    merged = pd.merge(
        df_tcfd,
        dqi[["bank_key", "year", "coverage", "specificity_ratio",
             "commitment_ratio", "dqi", "dqi_alt"]],
        on=["bank_key", "year"],
        suffixes=("_tcfd", "_dqi"),
    )

    print(f"  ✓ Merged {len(merged)} rows for correlation analysis")

    # Diagnostic if merge still fails
    if len(merged) == 0:
        tcfd_keys = sorted(df_tcfd["bank_key"].unique())
        dqi_keys  = sorted(dqi["bank_key"].unique())
        missing   = set(tcfd_keys) - set(dqi_keys)
        print("  ✗ No rows matched. Check key mismatches:")
        print(f"    TCFD keys  (sample): {tcfd_keys[:5]}")
        print(f"    DQI  keys  (sample): {dqi_keys[:5]}")
        if missing:
            print(f"    Keys in TCFD but not DQI: {sorted(missing)}")
        raise ValueError(
            "Merge produced 0 rows — bank keys don't align between "
            "TCFD files and DQI file even after normalisation. "
            "See diagnostic output above."
        )

    rows = []
    for pillar_col, pillar_label in [
        ("pct_governance", "Governance"),
        ("pct_strategy",   "Strategy"),
        ("pct_risk",       "Risk management"),
        ("pct_metrics",    "Metrics / targets"),
        ("balance",        "Pillar balance"),
    ]:
        for dqi_col, dqi_label in [
            ("specificity_ratio", "Specificity ratio"),
            ("commitment_ratio",  "Commitment ratio"),
            ("dqi",               "DQI (z-score)"),
            ("dqi_alt",           "DQI (raw)"),
        ]:
            r, p = stats.pearsonr(merged[pillar_col], merged[dqi_col])
            rows.append({
                "TCFD metric":   pillar_label,
                "DQI component": dqi_label,
                "N":             len(merged),
                "Pearson r":     round(r, 4),
                "p-value":       round(p, 4),
                "Stars": (
                    "***" if p < 0.01 else
                    "**"  if p < 0.05 else
                    "*"   if p < 0.10 else ""
                ),
            })
    return pd.DataFrame(rows)


# ── Step 4: Per-bank deltas ────────────────────────────────────────────────────

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 2024 − 2023 change in each pillar share per bank.
    """
    d23 = df[df["year"] == 2023].set_index("bank")
    d24 = df[df["year"] == 2024].set_index("bank")
    common = sorted(d23.index.intersection(d24.index))

    rows = []
    for bank in common:
        row = {"bank": bank}
        for p in PILLARS:
            col = f"pct_{p}"
            row[f"{p}_2023"]  = round(d23.loc[bank, col], 4)
            row[f"{p}_2024"]  = round(d24.loc[bank, col], 4)
            row[f"delta_{p}"] = round(d24.loc[bank, col] - d23.loc[bank, col], 4)
        row["balance_2023"]  = round(d23.loc[bank, "balance"], 4)
        row["balance_2024"]  = round(d24.loc[bank, "balance"], 4)
        row["delta_balance"] = round(d24.loc[bank, "balance"] - d23.loc[bank, "balance"], 4)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("delta_metrics", ascending=False)


# ── Step 5: Print summaries ────────────────────────────────────────────────────

def print_summary(df_tcfd, df_avg, df_corr, df_delta):
    SEP = "=" * 65

    print("\n" + SEP)
    print("  TCFD Pillar Distribution — Sample Overview")
    print(SEP)
    print(f"  Bank-years : {len(df_tcfd)}  |  Banks : {df_tcfd['bank'].nunique()}")
    print(f"  Dominant pillar: strategy in "
          f"{(df_tcfd['dominant_pillar']=='strategy').sum()}/{len(df_tcfd)} bank-years\n")

    print("  Mean pillar shares by year:")
    for yr in [2023, 2024]:
        sub   = df_tcfd[df_tcfd["year"] == yr]
        parts = "  |  ".join(
            f"{p}: {sub[f'pct_{p}'].mean():.1%}" for p in PILLARS
        )
        print(f"    {yr}: {parts}")

    print("\n" + SEP)
    print("  Paired t-tests: 2024 vs 2023")
    print(SEP)
    print(f"  {'Metric':<25} {'Mean23':>7} {'Mean24':>7} "
          f"{'Delta':>8} {'t':>7} {'p':>8} {'Sig'}")
    print(f"  {'-'*65}")
    for _, r in df_avg.iterrows():
        print(f"  {r['Metric']:<25} {r['Mean 2023']:>7.4f} {r['Mean 2024']:>7.4f} "
              f"{r['Delta']:>+8.4f} {r['t-stat']:>7.3f} {r['p-value']:>8.4f} "
              f"{r['Stars']}")

    print("\n" + SEP)
    print("  Key Correlations: pillar shares vs DQI")
    print(SEP)
    focus = df_corr[df_corr["DQI component"] == "DQI (z-score)"]
    for _, r in focus.iterrows():
        print(f"  {r['TCFD metric']:<22} vs DQI: "
              f"r = {r['Pearson r']:+.3f}  p = {r['p-value']:.3f}  {r['Stars']}")

    print("\n" + SEP)
    print("  Top 5 metrics/targets increasers (Δ 2023→2024)")
    print(SEP)
    for _, r in df_delta.head(5).iterrows():
        print(f"  {r['bank']:<25}  Δmetrics = {r['delta_metrics']:+.3f}")

    print("\n  Top 5 metrics/targets decliners:")
    for _, r in df_delta.tail(5).iterrows():
        print(f"  {r['bank']:<25}  Δmetrics = {r['delta_metrics']:+.3f}")


# ── Step 6: Save ───────────────────────────────────────────────────────────────

def save_results(df_tcfd, df_avg, df_corr, df_delta, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    today    = date.today().isoformat()
    out_path = os.path.join(out_dir, f"tcfd_analysis_{today}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_tcfd.to_excel(writer, sheet_name="PillarSummary",  index=False)
        df_avg.to_excel(writer,  sheet_name="SampleAverages", index=False)
        df_corr.to_excel(writer, sheet_name="Correlations",   index=False)
        df_delta.to_excel(writer,sheet_name="Deltas",         index=False)

    print(f"\n✓ Saved to: {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TCFD descriptive analysis.")
    parser.add_argument("--tcfd_dir", default=TCFD_DIR,
                        help="Folder containing TCFD output .xlsx files")
    parser.add_argument("--dqi",      default=None,
                        help="Path to DQI file (auto-detects latest if omitted)")
    args = parser.parse_args()

    # Auto-detect latest DQI file
    if args.dqi:
        dqi_path = args.dqi
    else:
        files = glob.glob(os.path.join(DQI_DIR, "dqi_*.xlsx"))
        if not files:
            raise FileNotFoundError(
                f"No DQI files found in {DQI_DIR}. "
                "Run build_dqi.py first, or pass --dqi <path>."
            )
        dqi_path = max(files, key=os.path.getmtime)

    print(f"\nTCFD folder : {args.tcfd_dir}")
    print(f"DQI file    : {dqi_path}\n")

    print("Loading TCFD files...")
    df_tcfd = load_tcfd_files(args.tcfd_dir)

    print("\nComputing sample averages and t-tests...")
    df_avg = compute_sample_averages(df_tcfd)

    print("\nComputing correlations with DQI...")
    df_corr = compute_correlations(df_tcfd, dqi_path)

    print("\nComputing per-bank deltas...")
    df_delta = compute_deltas(df_tcfd)

    print_summary(df_tcfd, df_avg, df_corr, df_delta)
    save_results(df_tcfd, df_avg, df_corr, df_delta, OUT_DIR)


if __name__ == "__main__":
    main()
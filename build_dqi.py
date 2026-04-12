"""
DQI Construction Script
-----------------------
Reads all specificity and commitment output files, matches them by
bank-year-report prefix, and constructs the Disclosure Quality Index (DQI).

Formula:
    DQI_i = mean( Z(Coverage_i), Z(Specificity_i), Z(Commitment_i) )

Where:
    Coverage_i      = climate paragraphs / total paragraphs
    Specificity_i   = specific climate paragraphs / climate paragraphs
    Commitment_i    = commitment climate paragraphs / climate paragraphs

Input folders:
    results/specificity/   — output of climate_specificity.py
    results/commitment/    — output of climate_commitment.py

Output:
    results/dqi/dqi_YYYY-MM-DD.xlsx   (one row per bank-year)

Run locally:
    python build_dqi.py

Optional arguments:
    python build_dqi.py --spec_dir results/specificity --comm_dir results/commitment --output results/dqi/my_dqi.xlsx
"""

import os
import argparse
import pandas as pd
from datetime import date


# ── Label config ──────────────────────────────────────────────────────────────
# These are the positive (presence) labels output by each ClimateBERT model.
# Adjust here if the model outputs different label strings.
SPECIFICITY_POSITIVE_LABEL = "spec"   # climatebert specificity: 'spec' = specific
COMMITMENT_POSITIVE_LABEL  = "yes"    # climatebert commitment: 'yes' = commitment/action present


# ── Step 1: File matching ─────────────────────────────────────────────────────

def extract_base(filename: str, stage: str) -> str:
    """
    Extract the base prefix from a pipeline output filename.

    Example:
        'UBS_2023_Annual_specificity_2026-04-12.xlsx', stage='specificity'
        → 'UBS_2023_Annual'
    """
    name = os.path.splitext(filename)[0]
    marker = f"_{stage}_"
    if marker in name:
        return name[: name.index(marker)]
    return None


def find_file_pairs(spec_dir: str, comm_dir: str) -> list:
    """
    Scan specificity and commitment folders and return matched (base, spec_path, comm_path) tuples.

    If multiple files exist for the same base (e.g. reruns on different dates),
    the most recent file is used (files are sorted in reverse alphabetical order,
    which works because dates are ISO format).

    Returns
    -------
    list of (base, spec_path, comm_path)
    """
    def index_folder(folder, stage):
        index = {}
        if not os.path.isdir(folder):
            return index
        for f in sorted(os.listdir(folder), reverse=True):   # latest date first
            if not f.endswith(".xlsx"):
                continue
            base = extract_base(f, stage)
            if base and base not in index:
                index[base] = os.path.join(folder, f)
        return index

    spec_index = index_folder(spec_dir, "specificity")
    comm_index = index_folder(comm_dir, "commitment")

    all_bases = sorted(set(spec_index) | set(comm_index))
    pairs = []

    for base in all_bases:
        has_spec = base in spec_index
        has_comm = base in comm_index
        if has_spec and has_comm:
            pairs.append((base, spec_index[base], comm_index[base]))
        elif has_spec:
            print(f"  ⚠  No commitment file found for: {base}  (skipped)")
        else:
            print(f"  ⚠  No specificity file found for: {base}  (skipped)")

    return pairs


# ── Step 2: Compute per-bank-year metrics ─────────────────────────────────────

def compute_metrics(base: str, spec_path: str, comm_path: str) -> dict:
    """
    Load a matched specificity/commitment pair, merge on paragraph_id,
    and compute raw DQI component metrics.

    Returns
    -------
    dict with bank, year, report_type, counts, and raw ratios
    """
    spec_df = pd.read_excel(spec_path)
    comm_df = pd.read_excel(comm_path)

    # Merge: take all columns from specificity file, add commitment columns from commitment file
    required_comm_cols = {"paragraph_id", "commitment_label", "commitment_score"}
    missing = required_comm_cols - set(comm_df.columns)
    if missing:
        raise ValueError(f"Commitment file '{os.path.basename(comm_path)}' is missing columns: {missing}")

    df = spec_df.merge(
        comm_df[["paragraph_id", "commitment_label", "commitment_score"]],
        on="paragraph_id",
        how="inner",
    )

    if df.empty:
        print(f"  ⚠  No matching paragraph_ids for: {base}  (skipped)")
        return None

    # ── Metadata ──────────────────────────────────────────────────────────────
    bank        = str(df["bank"].iloc[0])        if "bank"        in df.columns else base
    year        = int(df["year"].iloc[0])        if "year"        in df.columns else None
    report_type = str(df["report_type"].iloc[0]) if "report_type" in df.columns else None

    # ── Coverage ──────────────────────────────────────────────────────────────
    total_paragraphs   = len(df)
    climate_paragraphs = int((df["detector_label"] == "yes").sum())
    coverage           = climate_paragraphs / total_paragraphs if total_paragraphs > 0 else 0.0

    # ── Specificity ratio (among climate paragraphs) ──────────────────────────
    climate_df         = df[df["detector_label"] == "yes"]
    specific_paragraphs = int((climate_df["specificity_label"] == SPECIFICITY_POSITIVE_LABEL).sum())
    specificity_ratio   = specific_paragraphs / climate_paragraphs if climate_paragraphs > 0 else 0.0

    # ── Commitment ratio (among climate paragraphs) ───────────────────────────
    commitment_paragraphs = int((climate_df["commitment_label"] == COMMITMENT_POSITIVE_LABEL).sum())
    commitment_ratio      = commitment_paragraphs / climate_paragraphs if climate_paragraphs > 0 else 0.0

    return {
        "bank":                    bank,
        "year":                    year,
        "report_type":             report_type,
        "total_paragraphs":        total_paragraphs,
        "climate_paragraphs":      climate_paragraphs,
        "coverage":                round(coverage, 4),
        "specific_paragraphs":     specific_paragraphs,
        "specificity_ratio":       round(specificity_ratio, 4),
        "commitment_paragraphs":   commitment_paragraphs,
        "commitment_ratio":        round(commitment_ratio, 4),
    }


# ── Step 3: Z-standardize and compute DQI ────────────────────────────────────

def zscore(series: pd.Series) -> pd.Series:
    """
    Compute z-scores for a series.
    Returns a zero series if standard deviation is 0 (only 1 observation or no variance).
    """
    std = series.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def build_dqi_table(metrics_list: list) -> pd.DataFrame:
    """
    Z-standardize Coverage, Specificity, Commitment and compute DQI.

    DQI_i = mean( Z(Coverage_i), Z(Specificity_i), Z(Commitment_i) )

    Returns
    -------
    pd.DataFrame with one row per bank-year
    """
    df = pd.DataFrame(metrics_list)

    if len(df) < 2:
        print(
            "\n  ⚠  Warning: Only 1 observation found. "
            "Z-scores default to 0. The DQI requires multiple banks to be meaningful.\n"
        )

    # Z-standardize each component
    df["z_coverage"]    = zscore(df["coverage"]).round(4)
    df["z_specificity"] = zscore(df["specificity_ratio"]).round(4)
    df["z_commitment"]  = zscore(df["commitment_ratio"]).round(4)

    # DQI = equally-weighted mean of z-scores (main specification)
    df["dqi"] = df[["z_coverage", "z_specificity", "z_commitment"]].mean(axis=1).round(4)

    # DQI_alt = raw average of unscaled ratios (robustness check)
    # Not sample-relative, so comparable across different runs / subsamples
    df["dqi_alt"] = df[["coverage", "specificity_ratio", "commitment_ratio"]].mean(axis=1).round(4)

    # ── Column order ──────────────────────────────────────────────────────────
    col_order = [
        "bank", "year", "report_type",
        "total_paragraphs", "climate_paragraphs", "coverage",
        "specific_paragraphs", "specificity_ratio",
        "commitment_paragraphs", "commitment_ratio",
        "z_coverage", "z_specificity", "z_commitment",
        "dqi", "dqi_alt",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    return df


# ── Step 4: Summary printout ──────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print a readable summary of DQI results to the console."""
    print("\n" + "=" * 65)
    print("  DQI Summary")
    print("=" * 65)
    print(f"  Bank-years processed : {len(df)}")
    print(f"  Coverage  — mean: {df['coverage'].mean():.3f}   std: {df['coverage'].std():.3f}")
    print(f"  Specificity ratio — mean: {df['specificity_ratio'].mean():.3f}   std: {df['specificity_ratio'].std():.3f}")
    print(f"  Commitment ratio  — mean: {df['commitment_ratio'].mean():.3f}   std: {df['commitment_ratio'].std():.3f}")
    print(f"  DQI     — mean: {df['dqi'].mean():.3f}   std: {df['dqi'].std():.3f}   "
          f"min: {df['dqi'].min():.3f}   max: {df['dqi'].max():.3f}")
    print(f"  DQI_alt — mean: {df['dqi_alt'].mean():.3f}   std: {df['dqi_alt'].std():.3f}   "
          f"min: {df['dqi_alt'].min():.3f}   max: {df['dqi_alt'].max():.3f}")
    print("=" * 65)

    print("\n  DQI Rankings (top to bottom):\n")
    ranked = df[["bank", "year", "report_type", "dqi"]].sort_values("dqi", ascending=False)
    for _, row in ranked.iterrows():
        print(f"    {row['bank']:<30} {int(row['year']) if pd.notna(row['year']) else '':>4}  "
              f"{str(row['report_type']):<15}  DQI = {row['dqi']:>7.4f}")
    print()


# ── Main entry point ──────────────────────────────────────────────────────────

def run_dqi_construction(
    spec_dir:    str = os.path.join("results", "specificity"),
    comm_dir:    str = os.path.join("results", "commitment"),
    output_path: str = None,
) -> tuple:
    """
    Full DQI pipeline: match files → merge → compute metrics → z-score → save.

    Parameters
    ----------
    spec_dir     : folder containing specificity output .xlsx files
    comm_dir     : folder containing commitment output .xlsx files
    output_path  : path to save DQI table; auto-generated if not provided

    Returns
    -------
    (df, output_path) : tuple
        df          — DQI table as a DataFrame
        output_path — path to saved Excel file
    """
    if output_path is None:
        today      = date.today().isoformat()
        output_dir = os.path.join("results", "dqi")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"dqi_{today}.xlsx")

    print(f"\nScanning for file pairs...")
    print(f"  Specificity folder : {spec_dir}")
    print(f"  Commitment folder  : {comm_dir}\n")

    pairs = find_file_pairs(spec_dir, comm_dir)

    if not pairs:
        raise FileNotFoundError(
            "No matching specificity/commitment file pairs found. "
            "Check that both folders contain output files with matching bank-year prefixes."
        )

    print(f"  ✓ {len(pairs)} matched pair(s) found\n")

    # ── Compute metrics for each bank-year ────────────────────────────────────
    metrics_list = []
    for base, spec_path, comm_path in pairs:
        print(f"  Processing: {base}")
        result = compute_metrics(base, spec_path, comm_path)
        if result is not None:
            metrics_list.append(result)

    if not metrics_list:
        raise ValueError("No metrics could be computed. Check your input files.")

    # ── Build DQI table ───────────────────────────────────────────────────────
    dqi_df = build_dqi_table(metrics_list)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dqi_df.to_excel(output_path, index=False)

    print(f"\n✓ DQI table saved to: {output_path}")
    print_summary(dqi_df)

    return dqi_df, output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Disclosure Quality Index (DQI).")
    parser.add_argument("--spec_dir",  default=os.path.join("results", "specificity"), help="Specificity results folder")
    parser.add_argument("--comm_dir",  default=os.path.join("results", "commitment"),  help="Commitment results folder")
    parser.add_argument("--output",    default=None, help="Output path (auto-generated if omitted)")
    args = parser.parse_args()

    run_dqi_construction(
        spec_dir    = args.spec_dir,
        comm_dir    = args.comm_dir,
        output_path = args.output,
    )

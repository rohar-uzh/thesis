"""
count_paragraphs.py
-------------------
Counts cleaned paragraphs across all detector output files.

Detector output files are one xlsx per bank-year, named like:
    UBS_2023_Annual_detected_2026-04-12.xlsx
    BNP-Paribas_2024_URD_detected_2026-04-13.xlsx

Each file contains every cleaned paragraph for that bank-year, tagged
with a climate label ('yes'/'no'). So len(file) = total cleaned paragraphs
prior to climate filtering, which is what Section 4.2 needs.

Usage (from repo root, in PowerShell):
    python count_paragraphs.py
    python count_paragraphs.py --dir results/detected
"""

import argparse
import re
from pathlib import Path
import pandas as pd


def parse_filename(stem: str):
    """Extract (bank, year, report_type) from a detector output filename stem.

    Pattern: <bank>_<year>_<report_type>_detected_<date>
    Examples:
        UBS_2023_Annual            -> ('UBS', 2023, 'Annual')
        BNP-Paribas_2024_URD       -> ('BNP-Paribas', 2024, 'URD')
        Credit-Agricole_2023_URD   -> ('Credit-Agricole', 2023, 'URD')
    """
    m = re.match(r"^(.+?)_(\d{4})_([^_]+)_detected_", stem)
    if not m:
        return None
    bank, year, report_type = m.group(1), int(m.group(2)), m.group(3)
    return bank, year, report_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default="results/detector",
        help="Directory containing detector output xlsx files",
    )
    args = ap.parse_args()

    in_dir = Path(args.dir)
    if not in_dir.exists():
        raise SystemExit(f"Directory not found: {in_dir.resolve()}")

    files = sorted(in_dir.glob("*_detected_*.xlsx"))
    if not files:
        raise SystemExit(f"No detector output files found in {in_dir.resolve()}")

    rows = []
    skipped = []
    for f in files:
        meta = parse_filename(f.stem)
        if meta is None:
            skipped.append(f.name)
            continue
        bank, year, report_type = meta
        n_rows = len(pd.read_excel(f))
        rows.append(
            dict(
                bank=bank,
                year=year,
                report_type=report_type,
                paragraphs=n_rows,
                file=f.name,
            )
        )

    df = pd.DataFrame(rows).sort_values(["bank", "year"]).reset_index(drop=True)

    # Guard against duplicate re-runs: keep only the latest file per (bank, year).
    # If duplicates exist, the latest filename wins because file dates are encoded
    # in the suffix; sorted() already orders them.
    dup_mask = df.duplicated(subset=["bank", "year"], keep="last")
    duplicates = df[df.duplicated(subset=["bank", "year"], keep=False)]
    df_unique = df[~dup_mask].reset_index(drop=True)

    print("=" * 70)
    print(f"DETECTOR OUTPUT SUMMARY  ({in_dir.resolve()})")
    print("=" * 70)
    print(f"Files scanned:          {len(files)}")
    print(f"Files parsed:           {len(df)}")
    if skipped:
        print(f"Files skipped (bad name): {len(skipped)}")
        for s in skipped:
            print(f"    {s}")
    if not duplicates.empty:
        print(f"\nDuplicate (bank, year) detected — keeping latest of each:")
        print(duplicates.to_string(index=False))
        print(f"After dedup: {len(df_unique)} unique bank-years")
    print()

    # Coverage check: should be 94 (47 banks × 2 years).
    n_banks = df_unique["bank"].nunique()
    n_years = df_unique["year"].nunique()
    print(f"Unique banks:           {n_banks}")
    print(f"Years covered:          {sorted(df_unique['year'].unique())}")
    print(f"Bank-years total:       {len(df_unique)}  (expected 94)")
    if len(df_unique) != 94:
        missing_pairs = set(
            (b, y) for b in df_unique["bank"].unique() for y in [2023, 2024]
        ) - set(zip(df_unique["bank"], df_unique["year"]))
        if missing_pairs:
            print(f"WARNING — missing bank-years: {sorted(missing_pairs)}")

    print()
    print("PARAGRAPH COUNTS")
    print("-" * 70)
    p = df_unique["paragraphs"]
    print(f"Total paragraphs:       {p.sum():>10,}")
    print(f"Mean per doc:           {p.mean():>10,.0f}")
    print(f"Median per doc:         {p.median():>10,.0f}")
    print(f"Min per doc:            {p.min():>10,}  ({df_unique.loc[p.idxmin(), 'bank']} {df_unique.loc[p.idxmin(), 'year']})")
    print(f"Max per doc:            {p.max():>10,}  ({df_unique.loc[p.idxmax(), 'bank']} {df_unique.loc[p.idxmax(), 'year']})")
    print(f"Std dev:                {p.std():>10,.0f}")

    print()
    print("BY YEAR")
    print("-" * 70)
    by_year = df_unique.groupby("year")["paragraphs"].agg(["count", "sum", "mean"])
    by_year.columns = ["n_docs", "total", "mean_per_doc"]
    print(by_year.to_string(float_format=lambda x: f"{x:,.0f}"))

    # Save a per-doc CSV for reference / appendix table
    out_csv = in_dir.parent / "paragraph_counts.csv"
    df_unique.to_csv(out_csv, index=False)
    print(f"\nPer-document counts written to: {out_csv}")


if __name__ == "__main__":
    main()

"""
Robustness Checks & H1 Test
-----------------------------
Extends run_regression.py with:

  1. Variance Inflation Factors (VIF) for all predictors
  2. Regression with year dummy + bank-clustered standard errors
  3. H1 paired t-test on raw DQI (dqi_alt) — 2023 vs 2024
  4. Component-level paired t-tests (coverage, specificity, commitment)

Inputs (same as run_regression.py):
    results/dqi/<latest>.xlsx     — output of build_dqi.py
    data/bank_data.xlsx           — Characteristics sheet

Output:
    results/regression/robustness_results_<date>.xlsx

Usage:
    python run_robustness.py
    python run_robustness.py --dqi results/dqi/dqi_2026-05-02.xlsx
    python run_robustness.py --chars data/bank_data.xlsx
"""

import os
import glob
import argparse
import warnings
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

DQI_DIR    = os.path.join("results", "dqi")
CHARS_PATH = os.path.join("data", "bank_data.xlsx")
OUT_DIR    = os.path.join("results", "regression")

# Note: DQI file and Characteristics sheet both use lowercase-hyphen bank keys
# (e.g. 'abn-amro', 'ubs-group') — no name mapping required.


# ── Step 1: Load & merge ───────────────────────────────────────────────────────

def load_and_merge(dqi_path: str, chars_path: str) -> pd.DataFrame:
    dqi = pd.read_excel(dqi_path)

    chars = pd.read_excel(chars_path, sheet_name="Characteristics", header=0)
    chars.columns = [c.split("\n")[0].strip() for c in chars.columns]

    merged = pd.merge(
        dqi,
        chars[["bank", "year", "total_assets", "roa", "cet1"]],
        on=["bank", "year"],
        how="inner",
    )
    merged["size"]       = np.log(merged["total_assets"])
    merged["year_dummy"] = (merged["year"] == 2024).astype(int)
    merged = merged.dropna(subset=["dqi", "dqi_alt", "size", "roa", "cet1"])

    print(f"✓ Loaded {len(merged)} observations, {merged['bank'].nunique()} banks")
    return merged


# ── Step 2: VIFs ──────────────────────────────────────────────────────────────

def compute_vifs(df: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(df[["size", "roa", "cet1"]])
    vif_df = pd.DataFrame({
        "Variable": ["Size (log TA)", "ROA", "CET1"],
        "VIF":      [
            variance_inflation_factor(X.values, i)
            for i in range(1, X.shape[1])   # skip constant — its VIF is always huge
        ],
    })
    vif_df["Flag"] = vif_df["VIF"].apply(
        lambda v: "SEVERE" if v > 10 else "CHECK" if v > 5 else "OK"
    )
    return vif_df


# ── Step 3: OLS helper ─────────────────────────────────────────────────────────

def run_ols(df: pd.DataFrame, dep_var: str, predictors: list,
            cov_type: str, label: str, cov_kwds: dict = None) -> dict:
    y = df[dep_var]
    X = sm.add_constant(df[predictors])

    model = sm.OLS(y, X)
    res   = model.fit(cov_type=cov_type, cov_kwds=cov_kwds or {})

    rows = []
    for var in res.params.index:
        stars = (
            "***" if res.pvalues[var] < 0.01 else
            "**"  if res.pvalues[var] < 0.05 else
            "*"   if res.pvalues[var] < 0.10 else ""
        )
        rows.append({
            "Label":       label,
            "Dep var":     dep_var,
            "SE type":     cov_type,
            "Variable":    var,
            "Coefficient": round(res.params[var], 6),
            "Std. Error":  round(res.bse[var], 6),
            "t-stat":      round(res.tvalues[var], 4),
            "p-value":     round(res.pvalues[var], 4),
            "CI lower":    round(res.conf_int().loc[var, 0], 4),
            "CI upper":    round(res.conf_int().loc[var, 1], 4),
            "Stars":       stars,
            "N":           int(res.nobs),
            "R2":          round(res.rsquared, 6),
            "Adj R2":      round(res.rsquared_adj, 6),
        })
    return {"label": label, "result": res, "rows": rows}


# ── Step 4: Paired t-tests ────────────────────────────────────────────────────

def paired_ttest(df: pd.DataFrame, col: str, label: str) -> dict:
    d23 = df[df["year"] == 2023].set_index("bank")[col]
    d24 = df[df["year"] == 2024].set_index("bank")[col]
    common = d23.index.intersection(d24.index)
    d23, d24 = d23.loc[common], d24.loc[common]
    deltas = d24 - d23

    t_stat, p_val = stats.ttest_rel(d24, d23)
    sem = deltas.sem()

    return {
        "Metric":       label,
        "N pairs":      len(common),
        "Mean 2023":    round(d23.mean(), 4),
        "Mean 2024":    round(d24.mean(), 4),
        "Mean delta":   round(deltas.mean(), 4),
        "Std delta":    round(deltas.std(), 4),
        "t-stat":       round(t_stat, 4),
        "p-value":      round(p_val, 4),
        "CI lower":     round(deltas.mean() - 1.96 * sem, 4),
        "CI upper":     round(deltas.mean() + 1.96 * sem, 4),
        "Stars":        (
            "***" if p_val < 0.01 else
            "**"  if p_val < 0.05 else
            "*"   if p_val < 0.10 else ""
        ),
    }


# ── Step 5: Print & save ───────────────────────────────────────────────────────

def print_vifs(vif_df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("  Variance Inflation Factors")
    print("=" * 50)
    print(f"  {'Variable':<20} {'VIF':>8}  {'Flag'}")
    print(f"  {'-'*40}")
    for _, row in vif_df.iterrows():
        print(f"  {row['Variable']:<20} {row['VIF']:>8.3f}  {row['Flag']}")
    print("  Rule: VIF > 5 = concern  |  > 10 = severe")


def print_reg(r: dict) -> None:
    res = r["result"]
    print(f"\n{'='*60}")
    print(f"  {r['label']}")
    print(f"  N={int(res.nobs)}  R²={res.rsquared:.4f}  Adj R²={res.rsquared_adj:.4f}")
    print(f"{'='*60}")
    print(f"  {'Variable':<20} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
    print(f"  {'-'*58}")
    for row in r["rows"]:
        print(f"  {row['Variable']:<20} {row['Coefficient']:>10.4f} "
              f"{row['Std. Error']:>10.4f} {row['t-stat']:>8.3f} "
              f"{row['p-value']:>8.3f} {row['Stars']}")


def print_ttests(ttest_results: list) -> None:
    print("\n" + "=" * 65)
    print("  Paired t-tests: 2024 vs 2023")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Mean 23':>8} {'Mean 24':>8} {'Delta':>8} "
          f"{'t':>7} {'p':>8} {'Sig'}")
    print(f"  {'-'*65}")
    for r in ttest_results:
        print(f"  {r['Metric']:<22} {r['Mean 2023']:>8.4f} {r['Mean 2024']:>8.4f} "
              f"{r['Mean delta']:>+8.4f} {r['t-stat']:>7.3f} {r['p-value']:>8.4f} "
              f"{r['Stars']}")


def save_results(vif_df, reg_rows, ttest_results, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    today    = date.today().isoformat()
    out_path = os.path.join(out_dir, f"robustness_results_{today}.xlsx")

    all_reg_rows = [row for r in reg_rows for row in r["rows"]]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        vif_df.to_excel(writer, sheet_name="VIF", index=False)
        pd.DataFrame(all_reg_rows).to_excel(writer, sheet_name="Regression", index=False)
        pd.DataFrame(ttest_results).to_excel(writer, sheet_name="Paired t-tests", index=False)

    print(f"\n✓ Saved to: {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Robustness checks for DQI thesis.")
    parser.add_argument("--dqi",   default=None,       help="Path to DQI file (auto-detects latest)")
    parser.add_argument("--chars", default=CHARS_PATH, help="Path to bank_data.xlsx")
    args = parser.parse_args()

    # Find DQI file
    if args.dqi:
        dqi_path = args.dqi
    else:
        files = glob.glob(os.path.join(DQI_DIR, "*.xlsx"))
        if not files:
            raise FileNotFoundError(f"No DQI files in {DQI_DIR}. Run build_dqi.py first.")
        dqi_path = max(files, key=os.path.getmtime)

    print(f"\nUsing DQI file  : {dqi_path}")
    print(f"Using chars file: {args.chars}\n")

    df = load_and_merge(dqi_path, args.chars)

    # ── 1. VIFs ───────────────────────────────────────────────────────────────
    vif_df = compute_vifs(df)
    print_vifs(vif_df)

    # ── 2. Regressions ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  Regressions with year dummy")
    print("=" * 60)

    preds_base  = ["size", "roa", "cet1"]
    preds_dummy = ["size", "roa", "cet1", "year_dummy"]
    cluster_kw  = {"groups": df["bank"]}

    reg_results = [
        run_ols(df, "dqi",     preds_dummy, "cluster", "(1) Main DQI    — clustered SE + year dummy", cluster_kw),
        run_ols(df, "dqi_alt", preds_dummy, "cluster", "(2) Raw DQI     — clustered SE + year dummy", cluster_kw),
        run_ols(df, "dqi",     preds_dummy, "HC3",     "(3) Main DQI    — HC3 + year dummy"),
        run_ols(df, "dqi",     preds_base,  "HC3",     "(4) Main DQI    — HC3, no year dummy (baseline)"),
    ]

    for r in reg_results:
        print_reg(r)

    print("\n  Notes:")
    print("  * p<0.10   ** p<0.05   *** p<0.01")
    print("  Clustered SEs: 47 bank clusters (same bank appears in 2023 and 2024).")
    print("  Year dummy = 1 for 2024. Negative coef → DQI declined in 2024 vs 2023.")

    # ── 3. Paired t-tests ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  H1: Paired t-tests — 2024 vs 2023  (n = 47 bank pairs)")
    print("=" * 65)

    ttest_results = [
        paired_ttest(df, "dqi",              "DQI (z-score)"),
        paired_ttest(df, "dqi_alt",          "Raw DQI (dqi_alt)"),
        paired_ttest(df, "coverage",         "Coverage ratio"),
        paired_ttest(df, "specificity_ratio","Specificity ratio"),
        paired_ttest(df, "commitment_ratio", "Commitment ratio"),
    ]
    print_ttests(ttest_results)

    print("\n  Interpretation:")
    for r in ttest_results:
        direction = "increased" if r["Mean delta"] > 0 else "declined"
        print(f"  {r['Metric']:<22}: {direction} by {abs(r['Mean delta']):.4f}  "
              f"(p={r['p-value']:.4f} {r['Stars']})")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(vif_df, reg_results, ttest_results, OUT_DIR)


if __name__ == "__main__":
    main()
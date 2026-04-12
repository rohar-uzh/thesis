"""
Cross-Sectional OLS Regression: DQI on Bank Characteristics
-------------------------------------------------------------
Regresses the Disclosure Quality Index (DQI) on observable bank
characteristics as specified in the thesis proposal:

    DQI_i = α + β1·Size_i + β2·ROA_i + β3·CET1_i + ε_i

where:
    Size  = log(Total Assets)   [main explanatory variable]
    ROA   = Return on Assets    [profitability control]
    CET1  = CET1 capital ratio  [capital adequacy control]

Runs two specifications:
    (1) Main:       dependent variable = dqi       (z-score composite)
    (2) Robustness: dependent variable = dqi_alt   (raw average)

Inputs:
    - DQI file:   results/dqi/<filename>.xlsx      (output of build_dqi.py)
    - Bank chars: data/bank_characteristics.xlsx   (manually prepared)

Output:
    - results/regression/regression_results_<date>.xlsx

Usage:
    python run_regression.py
    python run_regression.py --dqi results/dqi/dqi_2026-04-12.xlsx
    python run_regression.py --chars data/bank_characteristics.xlsx
"""

import os
import sys
import argparse
import glob
import warnings
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

DQI_DIR   = os.path.join("results", "dqi")
CHARS_PATH = os.path.join("data", "bank_characteristics.xlsx")
OUT_DIR   = os.path.join("results", "regression")

# Column names expected in DQI file (from build_dqi.py)
DQI_COL     = "dqi"
DQI_ALT_COL = "dqi_alt"
BANK_COL    = "bank"
YEAR_COL    = "year"

# Column names expected in bank characteristics file
CHARS_COLS = {
    "bank":         "bank",           # must match DQI file bank names exactly
    "year":         "year",           # 4-digit integer
    "total_assets": "total_assets",   # in millions (or any consistent unit)
    "roa":          "roa",            # decimal (e.g. 0.012 for 1.2%)
    "cet1":         "cet1",           # decimal (e.g. 0.145 for 14.5%)
}


# ── Step 1: Find most recent DQI file ────────────────────────────────────────

def find_dqi_file(dqi_dir: str = DQI_DIR) -> str:
    """Return the most recently modified DQI Excel file in the results/dqi/ folder."""
    pattern = os.path.join(dqi_dir, "*.xlsx")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No .xlsx files found in '{dqi_dir}'.\n"
            f"Run build_dqi.py first to generate the DQI file."
        )
    latest = max(files, key=os.path.getmtime)
    return latest


# ── Step 2: Load & merge data ─────────────────────────────────────────────────

def load_dqi(filepath: str) -> pd.DataFrame:
    """Load DQI file and validate required columns."""
    df = pd.read_excel(filepath)
    required = {BANK_COL, YEAR_COL, DQI_COL, DQI_ALT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DQI file is missing columns: {missing}\n"
            f"Expected output from build_dqi.py with columns: {required}"
        )
    print(f"✓ Loaded DQI file: {os.path.basename(filepath)}")
    print(f"  Rows: {len(df):,}  |  Banks: {df[BANK_COL].nunique()}  |  "
          f"Years: {sorted(df[YEAR_COL].unique())}")
    print()
    return df


def load_characteristics(filepath: str) -> pd.DataFrame:
    """Load and validate bank characteristics file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Bank characteristics file not found: '{filepath}'\n"
            f"Run:  python run_regression.py --create-template\n"
            f"to generate a blank template at that path."
        )
    df = pd.read_excel(filepath, header=4)
    required = set(CHARS_COLS.values())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Bank characteristics file is missing columns: {missing}\n"
            f"Expected columns: {list(required)}"
        )
    print(f"✓ Loaded bank characteristics: {os.path.basename(filepath)}")
    print(f"  Rows: {len(df):,}  |  Banks: {df['bank'].nunique()}")
    print()
    return df


def merge_data(dqi_df: pd.DataFrame, chars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge DQI and bank characteristics on (bank, year).
    Adds log(total_assets) as the Size variable.
    Drops rows with missing values in any regression variable.
    """
    merged = pd.merge(
        dqi_df[[BANK_COL, YEAR_COL, DQI_COL, DQI_ALT_COL]],
        chars_df[["bank", "year", "total_assets", "roa", "cet1"]],
        on=["bank", "year"],
        how="inner",
    )

    n_dqi  = len(dqi_df)
    n_merged = len(merged)

    if n_merged == 0:
        raise ValueError(
            "Merge produced zero rows. Check that bank names and years "
            "match exactly between the DQI file and the characteristics file."
        )

    # Size = log(total assets)
    merged["size"] = np.log(merged["total_assets"])

    # Drop rows with any missing regression variable
    reg_cols = [DQI_COL, DQI_ALT_COL, "size", "roa", "cet1"]
    before = len(merged)
    merged = merged.dropna(subset=reg_cols)
    dropped = before - len(merged)

    print(f"✓ Merged dataset: {n_merged} rows matched "
          f"({n_dqi - n_merged} DQI rows unmatched in characteristics file)")
    if dropped:
        print(f"  Dropped {dropped} rows with missing values in regression variables")
    print(f"  Final regression sample: {len(merged)} observations, "
          f"{merged[BANK_COL].nunique()} banks")
    print()

    return merged


# ── Step 3: OLS regression ────────────────────────────────────────────────────

def run_ols(
    df: pd.DataFrame,
    dep_var: str,
    label: str,
) -> dict:
    """
    Run OLS:  dep_var = α + β1·size + β2·roa + β3·cet1 + ε

    Returns a dict with model results and diagnostics.
    """
    y = df[dep_var]
    X = sm.add_constant(df[["size", "roa", "cet1"]])
    X.columns = ["const", "Size (log TA)", "ROA", "CET1"]

    model = sm.OLS(y, X).fit(cov_type="HC3")   # heteroskedasticity-robust SEs

    # Breusch-Pagan test for heteroskedasticity
    bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)

    # Durbin-Watson (cross-section — informational only)
    dw = durbin_watson(model.resid)

    print(f"{'='*60}")
    print(f"  Specification: {label}")
    print(f"  Dependent variable: {dep_var}")
    print(f"  N = {int(model.nobs)}   R² = {model.rsquared:.4f}   "
          f"Adj. R² = {model.rsquared_adj:.4f}")
    print(f"{'='*60}")
    print(model.summary2(float_format="%.4f"))
    print(f"  Breusch-Pagan test  stat={bp_stat:.3f}  p={bp_pval:.3f}")
    print(f"  Durbin-Watson              {dw:.3f}")
    print()

    return {
        "label":    label,
        "dep_var":  dep_var,
        "model":    model,
        "bp_stat":  bp_stat,
        "bp_pval":  bp_pval,
        "dw":       dw,
    }


# ── Step 4: Format results table ─────────────────────────────────────────────

def build_results_table(results_list: list) -> pd.DataFrame:
    """
    Build a side-by-side coefficient table across specifications.
    Format: coef (std err) with significance stars.
    """
    rows = []

    var_labels = {
        "const":          "Constant",
        "Size (log TA)":  "Size (log Total Assets)",
        "ROA":            "ROA",
        "CET1":           "CET1 Ratio",
    }

    for res in results_list:
        m = res["model"]
        col_name = res["label"]

        for var in m.params.index:
            coef  = m.params[var]
            se    = m.bse[var]
            pval  = m.pvalues[var]
            stars = _stars(pval)
            rows.append({
                "Variable":      var_labels.get(var, var),
                "Specification": col_name,
                "Coefficient":   f"{coef:.4f}{stars}",
                "Std. Error":    f"({se:.4f})",
                "p-value":       f"{pval:.4f}",
            })

    # Pivot to wide format (one column per specification)
    coef_wide = (
        pd.DataFrame(rows)
        .pivot(index="Variable", columns="Specification", values="Coefficient")
    )
    se_wide = (
        pd.DataFrame(rows)
        .pivot(index="Variable", columns="Specification", values="Std. Error")
    )

    # Interleave coefficient and SE rows
    out_rows = []
    for var in var_labels.values():
        if var in coef_wide.index:
            coef_row = coef_wide.loc[var].to_dict()
            coef_row["Variable"] = var
            se_row = se_wide.loc[var].to_dict()
            se_row["Variable"] = ""
            out_rows.append(coef_row)
            out_rows.append(se_row)

    table = pd.DataFrame(out_rows).set_index("Variable")

    # Append fit statistics
    fit_rows = []
    for res in results_list:
        m = res["model"]
        fit_rows.append({
            "Specification": res["label"],
            "N":             int(m.nobs),
            "R²":            f"{m.rsquared:.4f}",
            "Adj. R²":       f"{m.rsquared_adj:.4f}",
            "BP p-value":    f"{res['bp_pval']:.3f}",
        })
    fit_df = (
        pd.DataFrame(fit_rows)
        .set_index("Specification")
        .T
    )
    fit_df.index.name = "Variable"

    return table, fit_df


def _stars(pval: float) -> str:
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.10:
        return "*"
    return ""


# ── Step 5: Save results ──────────────────────────────────────────────────────

def save_results(
    merged_df: pd.DataFrame,
    results_list: list,
    coef_table: pd.DataFrame,
    fit_table: pd.DataFrame,
    out_dir: str = OUT_DIR,
) -> str:
    """Save regression output to Excel with multiple sheets."""
    os.makedirs(out_dir, exist_ok=True)
    today = date.today().isoformat()
    out_path = os.path.join(out_dir, f"regression_results_{today}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # Sheet 1: Coefficient table
        coef_table.to_excel(writer, sheet_name="Coefficients")
        fit_table.to_excel(writer, sheet_name="Coefficients", startrow=len(coef_table) + 3)

        # Sheet 2: Full statsmodels summary per specification
        row = 0
        for res in results_list:
            summary_df = _summary_to_df(res["model"], res["label"])
            summary_df.to_excel(writer, sheet_name="Full Summary", startrow=row, index=False)
            row += len(summary_df) + 3

        # Sheet 3: Regression sample data
        merged_df.to_excel(writer, sheet_name="Regression Data", index=False)

    print(f"✓ Results saved to: {out_path}")
    return out_path


def _summary_to_df(model, label: str) -> pd.DataFrame:
    """Convert statsmodels OLS result to a flat DataFrame for Excel export."""
    rows = [{"Item": f"=== {label} ===", "Value": ""}]
    rows.append({"Item": "N",       "Value": int(model.nobs)})
    rows.append({"Item": "R²",      "Value": round(model.rsquared, 6)})
    rows.append({"Item": "Adj. R²", "Value": round(model.rsquared_adj, 6)})
    rows.append({"Item": "F-stat",  "Value": round(model.fvalue, 4)})
    rows.append({"Item": "F p-val", "Value": round(model.f_pvalue, 4)})
    rows.append({"Item": "--- Coefficients ---", "Value": ""})
    for var in model.params.index:
        rows.append({
            "Item":  var,
            "Value": (f"coef={model.params[var]:.6f}  "
                      f"se={model.bse[var]:.6f}  "
                      f"t={model.tvalues[var]:.4f}  "
                      f"p={model.pvalues[var]:.4f}"
                      f"  [{model.conf_int().loc[var,0]:.4f}, "
                      f"{model.conf_int().loc[var,1]:.4f}]"),
        })
    return pd.DataFrame(rows)


# ── Template generator ────────────────────────────────────────────────────────

def create_characteristics_template(out_path: str = CHARS_PATH):
    """
    Generate a blank bank characteristics Excel template.
    Includes column headers and a few example rows.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    example_data = {
        "bank":         ["UBS", "UBS", "HSBC", "HSBC", "BNP Paribas", "BNP Paribas"],
        "year":         [2023,   2024,  2023,   2024,   2023,           2024],
        "total_assets": [1700000, 1750000, 3000000, 3100000, 2900000, 2950000],
        # ↑ in millions of EUR/USD — use a consistent currency across all banks
        "roa":  [0.012, 0.013, 0.009, 0.010, 0.008, 0.009],
        # ↑ Return on Assets as a decimal (e.g. 0.012 = 1.2%)
        "cet1": [0.145, 0.147, 0.148, 0.150, 0.135, 0.138],
        # ↑ CET1 ratio as a decimal (e.g. 0.145 = 14.5%)
    }

    df = pd.DataFrame(example_data)
    df.to_excel(out_path, index=False)

    print(f"✓ Template created: {out_path}")
    print()
    print("  Fill in your bank data:")
    print("    bank         — must match bank names in the DQI file exactly")
    print("    year         — 4-digit year (2023 or 2024)")
    print("    total_assets — in millions, consistent currency across all banks")
    print("    roa          — as decimal (e.g. 0.012 for 1.2%)")
    print("    cet1         — as decimal (e.g. 0.145 for 14.5%)")
    print()
    print("  Data source: Bloomberg, bank annual reports, or ECB SDW.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-sectional OLS regression of DQI on bank characteristics."
    )
    parser.add_argument(
        "--dqi", default=None,
        help="Path to DQI Excel file. Auto-detects most recent file in results/dqi/ if omitted."
    )
    parser.add_argument(
        "--chars", default=CHARS_PATH,
        help=f"Path to bank characteristics Excel. Default: {CHARS_PATH}"
    )
    parser.add_argument(
        "--create-template", action="store_true",
        help=f"Create a blank bank characteristics template at --chars path and exit."
    )
    args = parser.parse_args()

    # ── Template mode ────────────────────────────────────────────────────────
    if args.create_template:
        create_characteristics_template(args.chars)
        sys.exit(0)

    # ── Normal mode ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  DQI Regression: Climate Disclosure Quality")
    print("=" * 60)
    print()

    # Load data
    dqi_path = args.dqi or find_dqi_file()
    dqi_df   = load_dqi(dqi_path)
    chars_df = load_characteristics(args.chars)
    merged   = merge_data(dqi_df, chars_df)

    # Run both specifications
    results = []
    results.append(run_ols(merged, DQI_COL,     label="(1) Main DQI (z-score)"))
    results.append(run_ols(merged, DQI_ALT_COL, label="(2) Robustness DQI (raw avg)"))

    # Build output table and save
    coef_table, fit_table = build_results_table(results)
    out_path = save_results(merged, results, coef_table, fit_table)

    print()
    print("--- Coefficient Table ---")
    print(coef_table.to_string())
    print()
    print("--- Fit Statistics ---")
    print(fit_table.to_string())
    print()
    print(f"Done. Results saved to: {out_path}")
    print()
    print("Notes:")
    print("  * p<0.10   ** p<0.05   *** p<0.01")
    print("  Standard errors are HC3 heteroskedasticity-robust.")
    print("  Size = log(Total Assets). DQI_alt = raw average (robustness).")


if __name__ == "__main__":
    main()

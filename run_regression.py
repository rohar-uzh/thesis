"""
Cross-Sectional OLS Regression: DQI on Bank Characteristics
-------------------------------------------------------------
Tests H2 (cross-sectional variation in DQI) with a three-spec build-up:

    Spec 1 (Baseline)    : DQI = α + β1·Size + β2·ROA + β3·CET1 + ε
    Spec 2 (+ ownership) : Spec 1 + γ1·D_Coop + γ2·D_State
    Spec 3 (+ country)   : Spec 2 + country-group dummies      [STUB — see TODO]

The Listed category is the omitted baseline for the ownership dummies.

The same three specs are mirrored with dqi_alt as the dependent variable on a
separate sheet, for robustness.

Standard errors: HC3 heteroskedasticity-robust throughout.

A joint F-test (Wald) on the ownership dummies is computed and reported
beneath the coefficient table — useful because individual t-stats may be
marginal even when the joint contribution is informative.

Inputs:
    - DQI file   : results/dqi/<file>.xlsx  (output of build_dqi.py)
                   Required cols: bank, year, dqi, dqi_alt
    - Bank chars : data/bank_data.xlsx, sheet "Characteristics"
                   Required cols: bank, year, total_assets, roa, cet1,
                                  ownership_type
                   Optional   : country, ownership_note

Output:
    - results/regression/regression_results_<date>.xlsx
        Sheet "Buildup (DQI)"     — 3-spec build-up on z-score DQI (main)
        Sheet "Buildup (alt)"     — same 3 specs on dqi_alt (robustness)
        Sheet "Full Summary"      — statsmodels detail per spec
        Sheet "Regression Data"   — merged dataset used for regression

Usage:
    python run_regression.py
    python run_regression.py --dqi results/dqi/dqi_2026-05-15.xlsx
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

DQI_DIR    = os.path.join("results", "dqi")
CHARS_PATH = os.path.join("data", "bank_data.xlsx")
OUT_DIR    = os.path.join("results", "regression")

# Column keys
DQI_COL     = "dqi"
DQI_ALT_COL = "dqi_alt"
BANK_COL    = "bank"
YEAR_COL    = "year"

# Ownership dummy levels. The first one is omitted (baseline category).
OWN_BASELINE = "Listed"
OWN_LEVELS   = ["Listed", "Cooperative", "StateOwned"]
OWN_DUMMIES  = [f"own_{lvl}" for lvl in OWN_LEVELS if lvl != OWN_BASELINE]
# → ["own_Cooperative", "own_StateOwned"]

# Country-group mapping (country name in Characteristics → group label)
COUNTRY_GROUP_MAP = {
    "Denmark":     "Nordic",
    "Finland":     "Nordic",
    "Norway":      "Nordic",
    "Sweden":      "Nordic",
    "Austria":     "Germanic",
    "Germany":     "Germanic",
    "Switzerland": "Germanic",
    "France":      "French",
    "Italy":       "Southern",
    "Spain":       "Southern",
    "UK":          "UkIe",
    "Ireland":     "UkIe",
    "Belgium":     "Benelux",
    "Netherlands": "Benelux",
}
# Germanic is omitted baseline (largest group: 11 banks, 22 obs)
COUNTRY_BASELINE = "Germanic"
COUNTRY_GROUPS   = ["Nordic", "French", "Southern", "UkIe", "Benelux"]
COUNTRY_DUMMIES  = [f"cg_{g}" for g in COUNTRY_GROUPS]
# → ["cg_Nordic", "cg_French", "cg_Southern", "cg_UkIe", "cg_Benelux"]

# Predictor sets for the three specs
PREDS_BASELINE  = ["size", "roa", "cet1"]
PREDS_OWNERSHIP = PREDS_BASELINE + OWN_DUMMIES
PREDS_COUNTRY   = PREDS_OWNERSHIP + COUNTRY_DUMMIES

# Variable display labels for the coefficient table
VAR_LABELS = {
    "const":           "Constant",
    "size":            "Size (log Total Assets)",
    "roa":             "ROA",
    "cet1":            "CET1 Ratio",
    "own_Cooperative": "D(Cooperative)",
    "own_StateOwned":  "D(State-Owned)",
    "cg_Nordic":       "D(Nordic)",
    "cg_French":       "D(French)",
    "cg_Southern":     "D(Southern)",
    "cg_UkIe":         "D(UK/IE)",
    "cg_Benelux":      "D(Benelux)",
}
# Order in which variables appear in the table
VAR_ORDER = [
    "const", "size", "roa", "cet1",
    "own_Cooperative", "own_StateOwned",
    "cg_Nordic", "cg_French", "cg_Southern", "cg_UkIe", "cg_Benelux",
]


# ── Step 1: Find DQI file ─────────────────────────────────────────────────────

def find_dqi_file(dqi_dir=DQI_DIR):
    pattern = os.path.join(dqi_dir, "*.xlsx")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No .xlsx files in '{dqi_dir}'. Run build_dqi.py first."
        )
    return max(files, key=os.path.getmtime)


# ── Step 2: Load data ─────────────────────────────────────────────────────────

def load_dqi(filepath):
    df = pd.read_excel(filepath)
    required = {BANK_COL, YEAR_COL, DQI_COL, DQI_ALT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DQI file missing columns: {missing}")
    print(f"✓ Loaded DQI: {os.path.basename(filepath)}")
    print(f"  {len(df):,} rows  |  {df[BANK_COL].nunique()} banks  |  "
          f"years {sorted(df[YEAR_COL].unique())}")
    print()
    return df


def load_characteristics(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")
    df = pd.read_excel(filepath, sheet_name="Characteristics", header=0)
    required = {"bank", "year", "total_assets", "roa", "cet1", "ownership_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Characteristics sheet missing columns: {missing}\n"
            f"Make sure ownership_type has been added (run the ownership "
            f"classification script first)."
        )
    # Verify ownership values
    levels = set(df["ownership_type"].dropna().unique())
    unknown = levels - set(OWN_LEVELS)
    if unknown:
        raise ValueError(
            f"Unknown ownership_type values: {unknown}. "
            f"Expected one of {OWN_LEVELS}."
        )
    print(f"✓ Loaded characteristics: {os.path.basename(filepath)}")
    print(f"  {len(df):,} rows  |  {df['bank'].nunique()} banks")
    own_dist = df.groupby("ownership_type").size().to_dict()
    print(f"  Ownership distribution: {own_dist}")
    print()
    return df


def merge_data(dqi_df, chars_df):
    """Merge DQI and characteristics; build size and ownership dummies."""
    chars_cols = ["bank", "year", "total_assets", "roa", "cet1", "ownership_type"]
    # Carry country forward if present (used by Spec 3 later)
    if "country" in chars_df.columns:
        chars_cols.append("country")

    merged = pd.merge(
        dqi_df[[BANK_COL, YEAR_COL, DQI_COL, DQI_ALT_COL]],
        chars_df[chars_cols],
        on=["bank", "year"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Merge produced 0 rows. Check bank/year alignment.")

    merged["size"] = np.log(merged["total_assets"])

    # Build ownership dummies. Listed = baseline (omitted).
    for lvl in OWN_LEVELS:
        if lvl == OWN_BASELINE:
            continue
        merged[f"own_{lvl}"] = (merged["ownership_type"] == lvl).astype(int)

    # Build country-group dummies. Germanic = baseline (omitted, 11 banks / 22 obs).
    merged["country_group"] = merged["country"].map(COUNTRY_GROUP_MAP)
    unmapped = merged[merged["country_group"].isna()]["country"].unique()
    if len(unmapped):
        raise ValueError(f"Unmapped countries: {list(unmapped)}. "
                         f"Update COUNTRY_GROUP_MAP in the config section.")
    for grp in COUNTRY_GROUPS:
        merged[f"cg_{grp}"] = (merged["country_group"] == grp).astype(int)

    reg_cols = [DQI_COL, DQI_ALT_COL] + PREDS_COUNTRY
    before = len(merged)
    merged = merged.dropna(subset=reg_cols)
    dropped = before - len(merged)

    print(f"✓ Merge: {len(merged)} rows matched")
    if dropped:
        print(f"  Dropped {dropped} rows with missing values")

    n_total = len(merged)
    n_coop  = int((merged["own_Cooperative"] == 1).sum())
    n_state = int((merged["own_StateOwned"]  == 1).sum())
    n_list  = n_total - n_coop - n_state
    cg_counts = {grp: int((merged[f"cg_{grp}"] == 1).sum()) for grp in COUNTRY_GROUPS}
    print(f"  Final sample: {n_total} obs, {merged[BANK_COL].nunique()} banks")
    print(f"  Ownership cells : Listed={n_list}, Cooperative={n_coop}, "
          f"StateOwned={n_state}")
    print(f"  Country groups  : Germanic={n_total - sum(cg_counts.values())} (baseline), "
          + ", ".join(f"{g}={v}" for g, v in cg_counts.items()))
    print()

    return merged


# ── Step 3: OLS runner ────────────────────────────────────────────────────────

def run_ols(df, dep_var, predictors, label):
    """
    Run OLS dep_var on predictors with HC3 SEs.
    Returns dict with model, diagnostics, and (if both ownership dummies are
    in predictors) the joint F-test on ownership.
    """
    y = df[dep_var]
    X = sm.add_constant(df[predictors])
    model = sm.OLS(y, X).fit(cov_type="HC3")

    # Diagnostics
    bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
    dw = durbin_watson(model.resid)

    # Joint F-test on ownership dummies, if both are in the spec
    own_test = None
    own_in_spec = [v for v in OWN_DUMMIES if v in predictors]
    if len(own_in_spec) >= 2:
        hypothesis = ", ".join(f"{v} = 0" for v in own_in_spec)
        test = model.wald_test(hypothesis, use_f=True)
        own_test = {
            "F":        float(np.squeeze(test.statistic)),
            "p":        float(np.squeeze(test.pvalue)),
            "df_num":   int(test.df_num),
            "df_denom": int(test.df_denom),
        }

    # Joint F-test on country dummies, if any are in the spec
    cg_test = None
    cg_in_spec = [v for v in COUNTRY_DUMMIES if v in predictors]
    if len(cg_in_spec) >= 1:
        hypothesis = ", ".join(f"{v} = 0" for v in cg_in_spec)
        test = model.wald_test(hypothesis, use_f=True)
        cg_test = {
            "F":        float(np.squeeze(test.statistic)),
            "p":        float(np.squeeze(test.pvalue)),
            "df_num":   int(test.df_num),
            "df_denom": int(test.df_denom),
        }

    return {
        "label":      label,
        "dep_var":    dep_var,
        "predictors": predictors,
        "model":      model,
        "bp_stat":    bp_stat,
        "bp_pval":    bp_pval,
        "dw":         dw,
        "own_test":   own_test,
        "cg_test":    cg_test,
    }


# ── Step 4: Console printing ──────────────────────────────────────────────────

def _stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:   return "***"
    if p < 0.05:   return "**"
    if p < 0.10:   return "*"
    return ""


def print_spec(res):
    m = res["model"]
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {res['label']}    [dep = {res['dep_var']}]")
    print(f"  N = {int(m.nobs)}   R² = {m.rsquared:.4f}   "
          f"Adj. R² = {m.rsquared_adj:.4f}")
    print(sep)
    print(f"  {'Variable':<28} {'Coef':>10}  {'HC3 SE':>10}  "
          f"{'t':>7}  {'p':>7}")
    print(f"  {'-'*68}")
    for v in m.params.index:
        label = VAR_LABELS.get(v, v)
        c = m.params[v]; se = m.bse[v]; t = m.tvalues[v]; p = m.pvalues[v]
        print(f"  {label:<28} {c:>10.4f}  {se:>10.4f}  {t:>7.3f}  "
              f"{p:>7.4f} {_stars(p)}")
    print(f"  {'-'*68}")
    if res["own_test"] is not None:
        t = res["own_test"]
        print(f"  Joint F-test ownership dummies : "
              f"F({t['df_num']}, {t['df_denom']}) = {t['F']:.3f}, "
              f"p = {t['p']:.4f} {_stars(t['p'])}")
    if res["cg_test"] is not None:
        t = res["cg_test"]
        print(f"  Joint F-test country dummies   : "
              f"F({t['df_num']}, {t['df_denom']}) = {t['F']:.3f}, "
              f"p = {t['p']:.4f} {_stars(t['p'])}")
    print(f"  Breusch-Pagan p = {res['bp_pval']:.3f}  |  DW = {res['dw']:.3f}")


# ── Step 5: Build-up table ────────────────────────────────────────────────────

def build_buildup_table(results_list):
    """
    Wide build-up table: columns = specifications, rows = variables.
    Each variable contributes two rows: coefficient (with stars) and SE in
    parentheses. Variables not in a given spec show '—'.

    Returns (coef_table_df, fit_table_df).
    """
    spec_labels = [r["label"] for r in results_list]

    # ---- Coefficient block ----
    coef_rows = []   # list of dicts: {Variable, spec1, spec2, spec3}

    for v in VAR_ORDER:
        # Skip variables that no spec uses (defensive)
        if not any(v in r["model"].params.index for r in results_list):
            continue

        coef_row = {"Variable": VAR_LABELS.get(v, v)}
        se_row   = {"Variable": ""}

        for r in results_list:
            m = r["model"]
            if v in m.params.index:
                c = m.params[v]; se = m.bse[v]; p = m.pvalues[v]
                coef_row[r["label"]] = f"{c:.4f}{_stars(p)}"
                se_row[r["label"]]   = f"({se:.4f})"
            else:
                coef_row[r["label"]] = "—"
                se_row[r["label"]]   = ""
        coef_rows.append(coef_row)
        coef_rows.append(se_row)

    coef_df = pd.DataFrame(coef_rows).set_index("Variable")

    # ---- Fit-stats block ----
    fit_rows = [
        {"Variable": "N",               **{r["label"]: int(r["model"].nobs)             for r in results_list}},
        {"Variable": "R²",              **{r["label"]: f"{r['model'].rsquared:.4f}"      for r in results_list}},
        {"Variable": "Adj. R²",         **{r["label"]: f"{r['model'].rsquared_adj:.4f}" for r in results_list}},
        {"Variable": "Breusch-Pagan p", **{r["label"]: f"{r['bp_pval']:.3f}"            for r in results_list}},
    ]
    own_row = {"Variable": "Ownership joint F-test (p)"}
    for r in results_list:
        own_row[r["label"]] = (f"{r['own_test']['p']:.4f}{_stars(r['own_test']['p'])}"
                               if r["own_test"] is not None else "—")
    fit_rows.append(own_row)

    cg_row = {"Variable": "Country joint F-test (p)"}
    for r in results_list:
        cg_row[r["label"]] = (f"{r['cg_test']['p']:.4f}{_stars(r['cg_test']['p'])}"
                              if r["cg_test"] is not None else "—")
    fit_rows.append(cg_row)

    fit_df = pd.DataFrame(fit_rows).set_index("Variable")

    return coef_df, fit_df


def print_buildup_table(coef_df, fit_df, title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(coef_df.to_string())
    print("  " + "-" * 76)
    print(fit_df.to_string())


# ── Step 6: Save to Excel ─────────────────────────────────────────────────────

def save_results(merged_df, results_main, results_alt,
                 coef_main, fit_main, coef_alt, fit_alt, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    today    = date.today().isoformat()
    out_path = os.path.join(out_dir, f"regression_results_{today}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Sheet 1: Build-up on z-score DQI (main)
        coef_main.to_excel(writer, sheet_name="Buildup (DQI)", startrow=1)
        fit_main.to_excel(writer, sheet_name="Buildup (DQI)",
                          startrow=len(coef_main) + 4)
        # Add title on row 0
        ws = writer.sheets["Buildup (DQI)"]
        ws.cell(row=1, column=1,
                value="Build-up regression — Main (dep var: dqi z-score)")

        # Sheet 2: Build-up on dqi_alt (robustness)
        coef_alt.to_excel(writer, sheet_name="Buildup (alt)", startrow=1)
        fit_alt.to_excel(writer, sheet_name="Buildup (alt)",
                         startrow=len(coef_alt) + 4)
        ws = writer.sheets["Buildup (alt)"]
        ws.cell(row=1, column=1,
                value="Build-up regression — Robustness (dep var: dqi_alt)")

        # Sheet 3: full statsmodels detail
        row = 0
        for r in results_main + results_alt:
            summary_df = _summary_to_df(r)
            summary_df.to_excel(writer, sheet_name="Full Summary",
                                startrow=row, index=False)
            row += len(summary_df) + 3

        # Sheet 4: regression data
        merged_df.to_excel(writer, sheet_name="Regression Data", index=False)

    print(f"\n✓ Results saved to: {out_path}")
    return out_path


def _summary_to_df(res):
    m = res["model"]
    rows = [{"Item": f"=== {res['label']}  (dep = {res['dep_var']}) ===",
             "Value": ""}]
    rows.append({"Item": "N",       "Value": int(m.nobs)})
    rows.append({"Item": "R²",      "Value": round(m.rsquared, 6)})
    rows.append({"Item": "Adj. R²", "Value": round(m.rsquared_adj, 6)})
    rows.append({"Item": "F-stat",  "Value": round(m.fvalue, 4)})
    rows.append({"Item": "F p-val", "Value": round(m.f_pvalue, 4)})
    rows.append({"Item": "BP p-val", "Value": round(res["bp_pval"], 4)})
    rows.append({"Item": "DW",       "Value": round(res["dw"], 4)})
    if res["own_test"]:
        t = res["own_test"]
        rows.append({"Item": "Ownership joint F",
                     "Value": f"F({t['df_num']},{t['df_denom']})={t['F']:.4f}, "
                              f"p={t['p']:.4f}"})
    rows.append({"Item": "--- Coefficients ---", "Value": ""})
    for v in m.params.index:
        rows.append({
            "Item":  VAR_LABELS.get(v, v),
            "Value": (f"coef={m.params[v]:.6f}  se={m.bse[v]:.6f}  "
                      f"t={m.tvalues[v]:.4f}  p={m.pvalues[v]:.4f}  "
                      f"[{m.conf_int().loc[v,0]:.4f}, "
                      f"{m.conf_int().loc[v,1]:.4f}]"),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def create_characteristics_template(out_path: str = CHARS_PATH):
    """Generate a blank bank characteristics template (ownership columns included)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    example = {
        "bank":           ["ubs-group", "ubs-group", "rabobank", "rabobank"],
        "country":        ["Switzerland", "Switzerland", "Netherlands", "Netherlands"],
        "year":           [2023, 2024, 2023, 2024],
        "total_assets":   [1551.8, 1512.3, 597.8, 612.1],   # EUR billions
        "roa":            [0.0194, 0.0031, 0.0042, 0.0045],  # decimal
        "cet1":           [0.143, 0.144, 0.188, 0.191],      # decimal
        "ownership_type": ["Listed", "Listed", "Cooperative", "Cooperative"],
        "ownership_note": ["", "", "Member-owned cooperative", "Member-owned cooperative"],
    }
    pd.DataFrame(example).to_excel(out_path, sheet_name="Characteristics", index=False)
    print(f"✓ Template created: {out_path}")
    print("  Fill in all 47 banks × 2 years = 94 rows.")
    print("  ownership_type: one of Listed / Cooperative / StateOwned")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-sectional OLS regression of DQI on bank characteristics "
                    "with three-spec build-up (baseline → +ownership → +country)."
    )
    parser.add_argument("--dqi",   default=None,       help="Path to DQI Excel file.")
    parser.add_argument("--chars", default=CHARS_PATH, help=f"Default: {CHARS_PATH}")
    parser.add_argument("--create-template", action="store_true",
                        help="Create a blank characteristics template and exit.")
    args = parser.parse_args()

    if args.create_template:
        create_characteristics_template(args.chars)
        import sys; sys.exit(0)

    print()
    print("=" * 70)
    print("  H2 Regression: DQI on Bank Characteristics — Three-Spec Build-up")
    print("=" * 70)
    print()

    dqi_path = args.dqi or find_dqi_file()
    dqi_df   = load_dqi(dqi_path)
    chars_df = load_characteristics(args.chars)
    merged   = merge_data(dqi_df, chars_df)

    # ── Spec definitions ─────────────────────────────────────────────────────
    # Additive build-up: each column adds variables to the previous spec.
    specs = [
        ("(1) Baseline",         PREDS_BASELINE),
        ("(2) + Ownership",      PREDS_OWNERSHIP),
        ("(3) + Country groups", PREDS_COUNTRY),
    ]

    # ── Main regressions on z-score DQI ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MAIN: dependent variable = dqi (z-score composite)")
    print("=" * 70)
    results_main = []
    for label, preds in specs:
        r = run_ols(merged, DQI_COL, preds, label)
        results_main.append(r)
        print_spec(r)

    coef_main, fit_main = build_buildup_table(results_main)
    print_buildup_table(coef_main, fit_main,
                        title="Main build-up — dep var: dqi (z-score)")

    # ── Robustness regressions on dqi_alt ────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  ROBUSTNESS: dependent variable = dqi_alt (raw average)")
    print("=" * 70)
    results_alt = []
    for label, preds in specs:
        r = run_ols(merged, DQI_ALT_COL, preds, label)
        results_alt.append(r)
        print_spec(r)

    coef_alt, fit_alt = build_buildup_table(results_alt)
    print_buildup_table(coef_alt, fit_alt,
                        title="Robustness build-up — dep var: dqi_alt")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = save_results(merged, results_main, results_alt,
                            coef_main, fit_main, coef_alt, fit_alt)

    print()
    print("Notes:")
    print("  * p<0.10   ** p<0.05   *** p<0.01")
    print("  Standard errors are HC3 heteroskedasticity-robust throughout.")
    print("  Baseline ownership category : Listed (omitted).")
    print("  Baseline country group      : Germanic (AT/CH/DE, 11 banks, 22 obs, omitted).")
    print("  Ownership cells : Cooperative=8 banks (16 obs), StateOwned=6 banks (12 obs).")
    print("  Country cells   : Nordic=8, French=6, Southern=9, UK/IE=8, Benelux=5 banks.")


if __name__ == "__main__":
    main()
"""
ClimateBERT Climate Specificity Classifier
------------------------------------------
Classifies climate-related paragraphs as specific or non-specific
using the ClimateBERT distilroberta-base-climate-specificity model.

Only runs on paragraphs already labelled climate-related by the detector
(detector_label == 'yes'). Non-climate rows are passed through unchanged.

Usage (from Colab notebook):
    from climate_specificity import run_specificity_classification
    df, output_file = run_specificity_classification("data/detected/UBS_2023_Annual_detected_2026-04-12.xlsx")
"""

import os
import pandas as pd
import torch
from datetime import date
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_NAME = "climatebert/distilroberta-base-climate-specificity"
BATCH_SIZE = 32


# ── Step 1: Load model ────────────────────────────────────────────────────────

def load_classifier():
    """
    Load the ClimateBERT specificity model.

    Returns
    -------
    clf : HuggingFace text-classification pipeline
    id2label : dict mapping raw labels to human names
    """
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    id2label = model.config.id2label

    print(f"✓ Model loaded: {MODEL_NAME}")
    print(f"  Device       : {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    print(f"  Label mapping: {id2label}")
    print()

    return clf, id2label


# ── Step 2: Load & validate data ──────────────────────────────────────────────

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Load the detector output Excel file and validate required columns.

    Expects columns: 'paragraph', 'detector_label'
    (i.e. the output of climate_detector.py)

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_excel(filepath)

    required_cols = {"paragraph", "detector_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}\n"
            f"Make sure you are passing the output of climate_detector.py."
        )

    df["paragraph"] = df["paragraph"].fillna("").astype(str).str.strip()

    n_climate = (df["detector_label"] == "yes").sum()
    n_total = len(df)

    print(f"✓ Loaded {n_total:,} paragraphs from: {os.path.basename(filepath)}")
    print(f"  Climate-related (to classify): {n_climate:,}")
    print(f"  Non-climate (skipped)        : {n_total - n_climate:,}")
    print()

    return df


# ── Step 3: Run inference ─────────────────────────────────────────────────────

def run_inference(clf, texts: list, batch_size: int = BATCH_SIZE) -> list:
    """
    Run the specificity classifier on a list of texts in batches.

    Returns
    -------
    list of dicts with 'label' and 'score' keys
    """
    all_results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying specificity"):
        batch = texts[i : i + batch_size]
        batch_results = clf(
            batch,
            truncation=True,
            padding=True,
            max_length=512,
            batch_size=batch_size,
        )
        all_results.extend(batch_results)

    return all_results


# ── Step 4: Generate output filename ──────────────────────────────────────────

def generate_output_filename(input_path: str) -> str:
    """
    Auto-generate a datestamped output filename.

    Example:
        input:  UBS_2023_Annual_detected_2026-04-12.xlsx
        output: results/specificity/UBS_2023_Annual_specificity_2026-04-12.xlsx
    """
    base = os.path.splitext(os.path.basename(input_path))[0]

    # Strip any existing stage suffix (_detected, _specificity, etc.)
    for suffix in ("_detected", "_specificity", "_commitment"):
        if suffix in base:
            base = base[: base.index(suffix)]

    # Strip trailing datestamp if present (_YYYY-MM-DD)
    if len(base) > 11 and base[-11] == "_":
        base = base[:-11]

    today = date.today().isoformat()
    output_dir = os.path.join("results", "specificity")
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, f"{base}_specificity_{today}.xlsx")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_specificity_classification(
    input_path: str,
    output_path: str = None,
    batch_size: int = BATCH_SIZE,
) -> tuple:
    """
    Full specificity pipeline: load → classify climate rows → save → return.

    Only paragraphs with detector_label == 'yes' are classified.
    All other rows receive NaN for specificity columns.

    Parameters
    ----------
    input_path : str
        Path to Excel file output by climate_detector.py.
    output_path : str, optional
        Where to save results. Auto-generated if not provided.
    batch_size : int
        Inference batch size (default 32).

    Returns
    -------
    (df, output_path) : tuple
        df          — Full DataFrame with added 'specificity_label' and 'specificity_score'
        output_path — Path where the Excel file was saved (for files.download())
    """
    if output_path is None:
        output_path = generate_output_filename(input_path)

    # Load and validate
    df = load_and_validate_data(input_path)
    clf, id2label = load_classifier()

    # Filter to climate-related paragraphs only
    climate_mask = df["detector_label"] == "yes"
    climate_texts = df.loc[climate_mask, "paragraph"].tolist()

    # Initialise output columns with NaN (non-climate rows stay NaN)
    df["specificity_label"] = None
    df["specificity_score"] = None

    # Run inference on climate rows only
    results = run_inference(clf, climate_texts, batch_size=batch_size)

    # Write results back into the correct rows
    df.loc[climate_mask, "specificity_label"] = [r["label"] for r in results]
    df.loc[climate_mask, "specificity_score"] = [round(r["score"], 4) for r in results]

    # ── Specificity summary ───────────────────────────────────────────────────
    climate_df = df[climate_mask]
    counts = climate_df["specificity_label"].value_counts()
    n_climate = len(climate_df)

    print("\n--- Specificity Summary (climate paragraphs only) ---")
    print(f"Label mapping: {id2label}")
    print()
    for label, count in counts.items():
        human = id2label.get(label, label)
        print(f"  {label} ({human:>12}): {count:>6,}  ({count/n_climate*100:.1f}%)")
    print()

    # Save full dataframe (all rows, NaN for non-climate)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"✓ Results saved to: {output_path}")

    return df, output_path


# ── CLI usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python climate_specificity.py <input.xlsx> [output.xlsx]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_specificity_classification(in_path, out_path)

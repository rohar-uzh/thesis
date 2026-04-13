"""
ClimateBERT Climate Detector Pipeline
--------------------------------------
Classifies paragraphs as climate-related or not using the
ClimateBERT distilroberta-base-climate-detector model.

Usage (from Colab notebook):
    from climate_detector import run_climate_detection
    df, output_file = run_climate_detection("data/my_paragraphs.xlsx")
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
MODEL_NAME = "climatebert/distilroberta-base-climate-detector"
BATCH_SIZE = 32


# ── Step 1: Load model ────────────────────────────────────────────────────────

def load_classifier():
    """
    Load the ClimateBERT climate-detector model.

    Returns
    -------
    clf : HuggingFace text-classification pipeline
    id2label : dict mapping raw labels (e.g. 'LABEL_0') to human names
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


# ── Step 2: Load & clean data ─────────────────────────────────────────────────

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load paragraph-level Excel file and strip empty rows.

    Expects a 'paragraph' column (output of pdf_parser.py).

    Returns
    -------
    pd.DataFrame with clean, non-empty paragraphs
    """
    df = pd.read_excel(filepath)
    df = df.copy()

    df["paragraph"] = df["paragraph"].fillna("").astype(str).str.strip()
    df = df[df["paragraph"] != ""].reset_index(drop=True)

    print(f"✓ Loaded {len(df):,} paragraphs from: {os.path.basename(filepath)}")
    print(f"  Columns: {df.columns.tolist()}")
    print()

    return df


# ── Step 3: Run inference ─────────────────────────────────────────────────────

def run_inference(clf, texts: list, batch_size: int = BATCH_SIZE) -> list:
    """
    Run the classifier on a list of texts in batches.

    Uses truncation=True so texts longer than 512 tokens are handled safely.

    Returns
    -------
    list of dicts, each with 'label' and 'score' keys
    """
    all_results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying paragraphs"):
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
    Auto-generate a datestamped output filename based on the input file.

    Example:
        input:  UBS_2023_Annual_2026-04-01.xlsx
        output: results/detector/UBS_2023_Annual_detected_2026-04-12.xlsx
    """
    base = os.path.splitext(os.path.basename(input_path))[0]
    # Remove any existing datestamp from parser output (last 11 chars if format _YYYY-MM-DD)
    if len(base) > 11 and base[-11] == "_":
        base = base[:-11]

    today = date.today().isoformat()
    output_dir = os.path.join("results", "detector")
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, f"{base}_detector_{today}.xlsx")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_climate_detection(
    input_path: str,
    output_path: str = None,
    batch_size: int = BATCH_SIZE,
) -> tuple:
    """
    Full detection pipeline: load → classify → save → return.

    Parameters
    ----------
    input_path : str
        Path to Excel file with a 'paragraph' column (output of pdf_parser.py).
    output_path : str, optional
        Where to save results. Auto-generated if not provided.
    batch_size : int
        Number of paragraphs per inference batch (default 32).

    Returns
    -------
    (df, output_path) : tuple
        df          — DataFrame with added 'detector_label' and 'detector_score' columns
        output_path — Path where the Excel file was saved (for files.download())
    """
    # Auto-generate output path if not provided
    if output_path is None:
        output_path = generate_output_filename(input_path)

    # Load data and model
    df = load_and_clean_data(input_path)
    clf, id2label = load_classifier()

    # Run inference
    texts = df["paragraph"].tolist()
    results = run_inference(clf, texts, batch_size=batch_size)

    # Attach results to dataframe
    df["detector_label"] = [r["label"] for r in results]
    df["detector_score"] = [round(r["score"], 4) for r in results]

    # ── Label distribution summary ────────────────────────────────────────────
    print("\n--- Detection Summary ---")
    print(f"Label mapping: {id2label}")
    print()
    counts = df["detector_label"].value_counts()
    total = len(df)
    for label, count in counts.items():
        human = id2label.get(label, label)
        print(f"  {label} ({human:>18}): {count:>6,}  ({count/total*100:.1f}%)")
    print()

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"✓ Results saved to: {output_path}")

    return df, output_path


# ── CLI usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python climate_detector.py <input.xlsx> [output.xlsx]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_climate_detection(in_path, out_path)

"""
ClimateBERT Climate Detector Pipeline
--------------------------------------
Runs the ClimateBERT climate-detection classifier on paragraph-level data.

Usage (from Colab notebook):
    from climate_detector import run_climate_detection
    df_results = run_climate_detection("data/paragraphs.xlsx")
"""

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


MODEL_NAME = "climatebert/distilroberta-base-climate-detector"
BATCH_SIZE = 32


def load_classifier():
    """Load ClimateBERT detector model and return a HuggingFace pipeline."""
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    print(f"Model loaded: {MODEL_NAME}")
    print(f"Device: {'GPU' if device == 0 else 'CPU'}")
    print(f"Label mapping: {model.config.id2label}")

    return clf, model.config.id2label


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load Excel file and clean paragraph column."""
    df = pd.read_excel(filepath)
    df = df.copy()
    df["paragraph"] = df["paragraph"].fillna("").astype(str).str.strip()
    df = df[df["paragraph"] != ""].reset_index(drop=True)

    print(f"Loaded {len(df)} paragraphs from {filepath}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def run_inference(clf, texts: list, batch_size: int = BATCH_SIZE) -> list:
    """Run classifier on a list of texts in batches."""
    all_results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
        batch = texts[i : i + batch_size]
        batch_results = clf(batch, truncation=True, padding=True, batch_size=batch_size)
        all_results.extend(batch_results)

    return all_results


def run_climate_detection(
    input_path: str,
    output_path: str = None,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Full pipeline: load data -> classify -> save results.

    Parameters
    ----------
    input_path : str
        Path to Excel file with a 'paragraph' column.
    output_path : str, optional
        Path to save results. If None, saves to 'results/detector_results.xlsx'.
    batch_size : int
        Inference batch size (default 32).

    Returns
    -------
    pd.DataFrame with added columns: detector_label, detector_score
    """
    if output_path is None:
        output_path = "results/detector_results.xlsx"

    # Load
    df = load_and_clean_data(input_path)
    clf, id2label = load_classifier()

    # Classify
    texts = df["paragraph"].tolist()
    results = run_inference(clf, texts, batch_size=batch_size)

    df["detector_label"] = [r["label"] for r in results]
    df["detector_score"] = [r["score"] for r in results]

    # Print label distribution for quick sanity check
    print("\n--- Label Distribution ---")
    print(df["detector_label"].value_counts())
    print(f"\nLabel mapping reminder: {id2label}")

    # Save
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python climate_detector.py <input.xlsx> [output.xlsx]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_climate_detection(in_path, out_path)

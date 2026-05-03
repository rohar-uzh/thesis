"""
climate_tcfd.py
---------------
Classifies climate-related paragraphs into the four TCFD recommendation
pillars using the ClimateBERT distilroberta-base-climate-tcfd model.

Model:  climatebert/distilroberta-base-climate-tcfd
Labels: governance | strategy | risk_management | metrics_targets

Input:  Detector output Excel file (output of climate_detector.py).
        Must contain 'detector_label' column.

Output: Same dataframe with two new columns added to every row:
          tcfd_label  — predicted TCFD pillar (climate rows only)
          tcfd_score  — model confidence score (0–1)

        Non-climate paragraphs (detector_label != 'yes') receive NaN for
        both new columns.

Note:   This model is trained on climate paragraphs only — the non-climate
        class was removed during fine-tuning. All climate paragraphs will
        receive one of the four TCFD pillar labels.

Output file is saved to:
    results/tcfd/<base>_tcfd_<YYYY-MM-DD>.xlsx

Run via the ClimateBERT TCFD Colab notebook, or import directly:
    from climate_tcfd import run_tcfd_classification
    df, output_file = run_tcfd_classification(input_path="path/to/file.xlsx")
"""

import os
import sys
import torch
import pandas as pd
from datetime import date
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "climatebert/distilroberta-base-climate-tcfd"

# Expected label strings produced by this model.
# Confirmed from the climatebert/tcfd_recommendations dataset label schema.
# Update here if the model config uses different string keys.
TCFD_LABELS = {
    0: "governance",
    1: "strategy",
    2: "risk_management",
    3: "metrics_targets",
}

DETECTOR_POSITIVE = "yes"   # value in detector_label that marks a climate paragraph


# ── Main entry point ──────────────────────────────────────────────────────────

def run_tcfd_classification(
    input_path: str,
    batch_size: int = 32,
    output_dir: str = None,
) -> tuple:
    """
    Classify climate paragraphs into TCFD pillars.

    Parameters
    ----------
    input_path : str
        Path to detector output Excel file.
    batch_size : int
        Batch size for GPU inference (default 32).
    output_dir : str, optional
        Folder for output file. Defaults to results/tcfd/ relative to repo root.

    Returns
    -------
    (df, output_file) : tuple
        df          — full DataFrame with tcfd_label and tcfd_score columns added
        output_file — path to saved Excel file
    """

    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_excel(input_path)

    required_cols = {"paragraph_id", "paragraph", "detector_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}\n"
            f"Make sure you are uploading the detector output Excel file."
        )

    # ── Derive output path ────────────────────────────────────────────────────
    base = _extract_base(input_path)
    today = date.today().isoformat()

    if output_dir is None:
        # Resolve relative to the thesis repo root (two levels up from this file)
        repo_root = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(repo_root, "results", "tcfd")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{base}_tcfd_{today}.xlsx")

    # ── Load model ────────────────────────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else -1
    device_name = f"GPU (cuda:{device})" if device >= 0 else "CPU"

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, max_len=512)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Build label mapping from the model's own config (authoritative source)
    id2label = model.config.id2label if hasattr(model.config, "id2label") else TCFD_LABELS
    print(f"\n✓ Model loaded: {MODEL_NAME}")
    print(f"  Device       : {device_name}")
    print(f"  Label mapping: {id2label}")
    print()

    if device >= 0:
        model = model.cuda()
    model.eval()

    # ── Filter to climate paragraphs ──────────────────────────────────────────
    climate_mask = df["detector_label"] == DETECTOR_POSITIVE
    climate_df   = df[climate_mask].copy()
    n_climate    = len(climate_df)
    n_total      = len(df)

    print(f"  Total paragraphs   : {n_total}")
    print(f"  Climate paragraphs : {n_climate}  ({100 * n_climate / n_total:.1f}%)")
    print(f"  Skipped (non-climate): {n_total - n_climate}")
    print()

    if n_climate == 0:
        print("⚠  No climate paragraphs found — nothing to classify.")
        df["tcfd_label"] = pd.NA
        df["tcfd_score"] = pd.NA
        df.to_excel(output_file, index=False)
        print(f"✓ Empty results saved to: {output_file}")
        return df, output_file

    # ── Run inference in batches ──────────────────────────────────────────────
    texts   = climate_df["paragraph"].fillna("").tolist()
    labels  = []
    scores  = []

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Classifying TCFD pillars",
    ):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        if device >= 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probs     = torch.softmax(logits, dim=-1)
        preds     = torch.argmax(probs, dim=-1).cpu().tolist()
        confidences = probs.max(dim=-1).values.cpu().tolist()

        labels.extend([id2label.get(p, str(p)) for p in preds])
        scores.extend([round(c, 4) for c in confidences])

    # ── Write results back to full dataframe ──────────────────────────────────
    df["tcfd_label"] = pd.NA
    df["tcfd_score"] = pd.NA

    df.loc[climate_mask, "tcfd_label"] = labels
    df.loc[climate_mask, "tcfd_score"] = scores

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_excel(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(df, climate_mask, id2label)

    return df, output_file


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_base(filepath: str) -> str:
    """
    Extract the base prefix from a detector output filename.

    Examples
    --------
    'UBS_2023_Annual_detector_2026-04-12.xlsx' → 'UBS_2023_Annual'
    '/content/thesis/data/detected/HSBC_2024_Annual_detector_2026-05-01.xlsx'
        → 'HSBC_2024_Annual'
    """
    name = os.path.splitext(os.path.basename(filepath))[0]
    for marker in ("_detector_", "_detected_"):
        if marker in name:
            return name[: name.index(marker)]
    # Fallback: use the full stem (handles re-upload of already-processed files)
    return name


def _print_summary(df: pd.DataFrame, climate_mask, id2label: dict) -> None:
    """Print TCFD pillar distribution for climate paragraphs."""
    climate_df  = df[climate_mask]
    n_climate   = len(climate_df)
    label_counts = climate_df["tcfd_label"].value_counts()

    print(f"\n--- TCFD Pillar Distribution (climate paragraphs only) ---")
    print(f"Label mapping: {id2label}\n")

    for label in id2label.values():
        count = label_counts.get(label, 0)
        pct   = 100 * count / n_climate if n_climate > 0 else 0.0
        print(f"  {label:<20}: {count:>5}  ({pct:.1f}%)")

    print()
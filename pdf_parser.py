"""
PDF Paragraph Extractor
-----------------------
Extracts paragraph-level text from bank annual/sustainability reports (PDF)
and outputs a clean Excel file ready for ClimateBERT classification.

Uses PyMuPDF's block extraction, where each "block" roughly corresponds
to a paragraph — which is exactly the unit of analysis ClimateBERT expects.

Usage (from Colab notebook):
    from pdf_parser import parse_pdf, parse_multiple_pdfs

    # Single PDF
    df = parse_pdf("data/ubs_annual_report_2024.pdf", bank_name="UBS", year=2024)

    # Multiple PDFs
    df = parse_multiple_pdfs("data/pdfs/")
"""

import pymupdf
import pandas as pd
import re
import os
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

# Minimum character length for a paragraph to be kept.
# Short fragments (headers, page numbers, captions) are noise for ClimateBERT.
MIN_PARAGRAPH_LENGTH = 50

# Maximum character length — extremely long blocks are likely parsing errors.
MAX_PARAGRAPH_LENGTH = 5000

# Patterns to remove: headers, footers, page numbers, boilerplate
BOILERPLATE_PATTERNS = [
    r"^\d+$",                          # Standalone page numbers
    r"^page\s+\d+",                    # "Page 1", "Page 23"
    r"^\d+\s*\|",                      # "1 |", "23 |" style page numbers
    r"^table of contents",             # Table of contents header
    r"©\s*\d{4}",                      # Copyright lines
    r"^annual report\s+\d{4}",         # Repeated report title headers
    r"^sustainability report\s+\d{4}", # Repeated report title headers
    r"www\.\S+\.\S+",                  # URLs (likely footers)
]


# ============================================================
# Core extraction functions
# ============================================================

def extract_paragraphs_from_page(page, page_num: int) -> list:
    """
    Extract text blocks (≈ paragraphs) from a single PDF page.

    Uses PyMuPDF's get_text("blocks") which returns a list of tuples:
        (x0, y0, x1, y1, "text content", block_no, type)
    where type=0 means text block, type=1 means image block.

    Parameters
    ----------
    page : pymupdf.Page
        A PyMuPDF page object.
    page_num : int
        1-based page number (for metadata).

    Returns
    -------
    list of dict
        Each dict has keys: paragraph, page_num, block_num, bbox
    """
    # Extract blocks, sort by reading order (top-to-bottom, left-to-right)
    blocks = page.get_text("blocks", sort=True)

    paragraphs = []
    for block in blocks:
        # block = (x0, y0, x1, y1, text, block_no, type)
        block_type = block[6]

        # Skip image blocks (type=1)
        if block_type != 0:
            continue

        text = block[4]

        # Clean the text
        text = clean_paragraph(text)

        # Skip empty or too-short paragraphs
        if len(text) < MIN_PARAGRAPH_LENGTH:
            continue

        # Skip too-long paragraphs (likely parsing errors)
        if len(text) > MAX_PARAGRAPH_LENGTH:
            continue

        # Skip boilerplate
        if is_boilerplate(text):
            continue

        paragraphs.append({
            "paragraph": text,
            "page_num": page_num,
            "block_num": block[5],
            "bbox": block[:4],  # Bounding box coordinates
        })

    return paragraphs


def clean_paragraph(text: str) -> str:
    """
    Clean a raw text block into a usable paragraph.

    Steps:
    1. Replace line breaks within the block with spaces
       (PDF blocks often have hard line breaks mid-sentence)
    2. Collapse multiple spaces
    3. Strip leading/trailing whitespace
    4. Remove common artifacts
    """
    # Replace newlines with spaces (PDF wraps lines within paragraphs)
    text = text.replace("\n", " ")

    # Remove hyphenation at line breaks (e.g., "sustain- ability" -> "sustainability")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    # Strip
    text = text.strip()

    return text


def is_boilerplate(text: str) -> bool:
    """
    Check if a paragraph matches common boilerplate patterns
    (headers, footers, page numbers, copyright notices).
    """
    text_lower = text.lower().strip()
    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


# ============================================================
# Main parsing functions
# ============================================================

def parse_pdf(
    pdf_path: str,
    bank_name: str = None,
    year: int = None,
    report_type: str = None,
) -> pd.DataFrame:
    """
    Parse a single PDF into a DataFrame of paragraphs.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    bank_name : str, optional
        Name of the bank (added as metadata column).
    year : int, optional
        Report year (added as metadata column).
    report_type : str, optional
        E.g. "annual_report" or "sustainability_report".

    Returns
    -------
    pd.DataFrame
        Columns: bank, year, report_type, page_num, paragraph
    """
    doc = pymupdf.open(pdf_path)

    all_paragraphs = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_paragraphs = extract_paragraphs_from_page(page, page_num + 1)
        all_paragraphs.extend(page_paragraphs)

    doc.close()

    # Build DataFrame
    df = pd.DataFrame(all_paragraphs)

    if df.empty:
        print(f"WARNING: No paragraphs extracted from {pdf_path}")
        return df

    # Add metadata columns
    df["bank"] = bank_name or Path(pdf_path).stem
    df["year"] = year
    df["report_type"] = report_type
    df["source_file"] = Path(pdf_path).name

    # Reorder columns for clarity
    cols = ["bank", "year", "report_type", "source_file", "page_num", "paragraph"]
    df = df[[c for c in cols if c in df.columns]]

    print(f"Extracted {len(df)} paragraphs from {Path(pdf_path).name}")

    return df


def parse_multiple_pdfs(
    pdf_folder: str,
    metadata_csv: str = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Parse all PDFs in a folder into a single combined DataFrame.

    Parameters
    ----------
    pdf_folder : str
        Path to folder containing PDF files.
    metadata_csv : str, optional
        Path to a CSV with columns: filename, bank, year, report_type
        If provided, metadata is matched to each PDF by filename.
        If not provided, bank name is inferred from filename.
    output_path : str, optional
        If provided, saves the combined DataFrame to this Excel path.

    Returns
    -------
    pd.DataFrame
        Combined paragraphs from all PDFs.
    """
    # Load metadata if provided
    metadata = None
    if metadata_csv and os.path.exists(metadata_csv):
        metadata = pd.read_csv(metadata_csv)
        print(f"Loaded metadata for {len(metadata)} files")

    # Find all PDFs
    pdf_files = sorted(Path(pdf_folder).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {pdf_folder}")

    all_dfs = []

    for pdf_path in pdf_files:
        # Look up metadata for this file
        bank_name = None
        year = None
        report_type = None

        if metadata is not None:
            match = metadata[metadata["filename"] == pdf_path.name]
            if not match.empty:
                row = match.iloc[0]
                bank_name = row.get("bank")
                year = row.get("year")
                report_type = row.get("report_type")

        df = parse_pdf(
            str(pdf_path),
            bank_name=bank_name,
            year=year,
            report_type=report_type,
        )

        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("WARNING: No paragraphs extracted from any PDF.")
        return pd.DataFrame()

    # Combine all
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} paragraphs from {len(all_dfs)} reports")

    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        combined.to_excel(output_path, index=False)
        print(f"Saved to: {output_path}")

    return combined


# ============================================================
# Quick test: run this file directly on a single PDF
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <input.pdf> [output.xlsx]")
        print("       python pdf_parser.py <pdf_folder/> [output.xlsx]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/parsed_paragraphs.xlsx"

    if os.path.isdir(input_path):
        parse_multiple_pdfs(input_path, output_path=output_path)
    else:
        df = parse_pdf(input_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_excel(output_path, index=False)
        print(f"Saved to: {output_path}")

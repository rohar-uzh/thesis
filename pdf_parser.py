"""
PDF Paragraph Extractor
-----------------------
Extracts paragraph-level text from bank annual/sustainability reports (PDF)
and outputs a clean Excel file ready for ClimateBERT classification.

Uses PyMuPDF's block extraction, where each "block" roughly corresponds
to a paragraph — which is exactly the unit of analysis ClimateBERT expects.

Output format (one row per paragraph):
    paragraph_id | bank | year | report_type | paragraph | word_count

Usage (from Colab notebook):
    from pdf_parser import parse_pdf, parse_multiple_pdfs

    # Single PDF
    df = parse_pdf(
        "data/ubs_sr_2024.pdf",
        bank_name="UBS",
        year=2024,
        report_type="Sustainability",
    )

    # Multiple PDFs in a folder (with metadata CSV)
    df = parse_multiple_pdfs("data/pdfs/", metadata_csv="data/pdf_metadata.csv")
"""

import pymupdf
import pandas as pd
import re
import os
from datetime import datetime
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

# Output directory for all parsed results
OUTPUT_DIR = "results/parsed"


def generate_output_filename(bank_name: str, year: int, output_dir: str = OUTPUT_DIR) -> str:
    """
    Generate a unique output filename in the format:
        results/parsed/BANKNAME_YEAR_YYYY-MM-DD.xlsx

    Example:
        results/parsed/UBS_2024_2026-03-29.xlsx

    Parameters
    ----------
    bank_name : str
        Name of the bank.
    year : int
        Report year.
    output_dir : str
        Directory to save in (default: results/parsed/).

    Returns
    -------
    str
        Full file path for the output Excel file.
    """
    # Clean bank name: lowercase, replace spaces with underscores
    clean_name = bank_name.lower().replace(" ", "_")
    # Current date in YYYY-MM-DD format
    today = datetime.now().strftime("%Y-%m-%d")
    # Build filename
    filename = f"{clean_name}_{year}_{today}.xlsx"
    return os.path.join(output_dir, filename)


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
        Each dict has keys: paragraph, page_num
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
        })

    return paragraphs


def clean_paragraph(text: str) -> str:
    """
    Clean a raw text block into a usable paragraph.

    Steps:
    1. Replace line breaks within the block with spaces
       (PDF blocks often have hard line breaks mid-sentence)
    2. Fix hyphenation at line breaks (e.g., "sustain- ability" → "sustainability")
    3. Collapse multiple spaces
    4. Strip leading/trailing whitespace
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


def generate_paragraph_id(bank_name: str, year: int, index: int) -> str:
    """
    Generate a unique paragraph ID in the format: BANK_YEAR_001

    Parameters
    ----------
    bank_name : str
        Bank name (spaces replaced with underscores, uppercased).
    year : int
        Report year.
    index : int
        1-based paragraph number.

    Returns
    -------
    str
        E.g. "UBS_2024_001", "HSBC_2023_042"
    """
    clean_name = bank_name.upper().replace(" ", "_")
    return f"{clean_name}_{year}_{index:03d}"


# ============================================================
# Main parsing functions
# ============================================================

def parse_pdf(
    pdf_path: str,
    bank_name: str,
    year: int,
    report_type: str = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Parse a single PDF into a DataFrame of paragraphs.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    bank_name : str
        Name of the bank (e.g. "UBS").
    year : int
        Report year (e.g. 2024).
    report_type : str, optional
        E.g. "Sustainability", "Annual", "Pillar 3".
    output_path : str, optional
        If provided, saves the DataFrame to this Excel path.

    Returns
    -------
    pd.DataFrame
        Columns: paragraph_id, bank, year, report_type, paragraph, word_count
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
    df["bank"] = bank_name
    df["year"] = year
    df["report_type"] = report_type

    # Generate paragraph IDs (1-based)
    df["paragraph_id"] = [
        generate_paragraph_id(bank_name, year, i + 1)
        for i in range(len(df))
    ]

    # Calculate word count
    df["word_count"] = df["paragraph"].str.split().str.len()

    # Reorder columns to match target format
    df = df[["paragraph_id", "bank", "year", "report_type", "paragraph", "word_count"]]

    print(f"Extracted {len(df)} paragraphs from {Path(pdf_path).name}")
    print(f"  Word count range: {df['word_count'].min()} – {df['word_count'].max()}")
    print(f"  Mean word count: {df['word_count'].mean():.0f}")

    # Auto-generate output path if not provided
    if output_path is None:
        output_path = generate_output_filename(bank_name, year)

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"  Saved to: {output_path}")

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
        Each row maps a PDF filename to its metadata.
    output_path : str, optional
        If provided, saves the combined DataFrame to this Excel path.

    Returns
    -------
    pd.DataFrame
        Combined paragraphs from all PDFs in target format.
    """
    # Load metadata
    if metadata_csv and os.path.exists(metadata_csv):
        metadata = pd.read_csv(metadata_csv)
        print(f"Loaded metadata for {len(metadata)} files")
    else:
        print("WARNING: No metadata CSV provided. Bank names will be inferred from filenames.")
        metadata = None

    # Find all PDFs
    pdf_files = sorted(Path(pdf_folder).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {pdf_folder}\n")

    all_dfs = []

    for pdf_path in pdf_files:
        # Look up metadata for this file
        bank_name = pdf_path.stem  # Default: use filename
        year = None
        report_type = None

        if metadata is not None:
            match = metadata[metadata["filename"] == pdf_path.name]
            if not match.empty:
                row = match.iloc[0]
                bank_name = row.get("bank", bank_name)
                year = row.get("year")
                report_type = row.get("report_type")
            else:
                print(f"  WARNING: No metadata found for {pdf_path.name}, skipping.")
                continue

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

    # Combine all DataFrames
    combined = pd.concat(all_dfs, ignore_index=True)

    # Re-generate unique paragraph IDs across the full dataset
    new_ids = []
    counters = {}
    for _, row in combined.iterrows():
        key = (row["bank"], row["year"])
        counters[key] = counters.get(key, 0) + 1
        new_ids.append(generate_paragraph_id(row["bank"], row["year"], counters[key]))
    combined["paragraph_id"] = new_ids

    print(f"\n{'='*50}")
    print(f"Total: {len(combined)} paragraphs from {len(all_dfs)} reports")
    print(f"Banks: {combined['bank'].nunique()}")
    print(f"{'='*50}")

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

    if len(sys.argv) < 4:
        print("Usage: python pdf_parser.py <input.pdf> <bank_name> <year> [report_type] [output.xlsx]")
        print("Example: python pdf_parser.py data/ubs_sr_2024.pdf UBS 2024 Sustainability")
        sys.exit(1)

    input_path = sys.argv[1]
    bank = sys.argv[2]
    yr = int(sys.argv[3])
    rtype = sys.argv[4] if len(sys.argv) > 4 else None
    out = sys.argv[5] if len(sys.argv) > 5 else "results/parsed_paragraphs.xlsx"

    df = parse_pdf(input_path, bank_name=bank, year=yr, report_type=rtype, output_path=out)

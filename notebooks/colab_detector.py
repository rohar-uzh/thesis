# =============================================================================
# Colab Notebook: ClimateBERT Climate Detection
# =============================================================================
# This notebook pulls the latest code from your GitHub repo and runs
# the climate detection pipeline on your uploaded data.
#
# Convert this to a .ipynb or just paste these cells into Colab.
# =============================================================================

# --- Cell 1: Setup & Install ---
# !pip install -q transformers datasets accelerate pandas openpyxl tqdm

# --- Cell 2: Clone or pull your repo ---
# First time:
# !git clone https://github.com/rohar-uzh/thesis.git /content/thesis
#
# Subsequent runs (pull latest changes):
# !cd /content/thesis && git pull

# --- Cell 3: Add repo to Python path ---
import sys
sys.path.insert(0, "/content/thesis")

# --- Cell 4: Upload your data ---
from google.colab import files
import shutil
import os

uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Move uploaded file into the repo's data folder
os.makedirs("/content/thesis/data", exist_ok=True)
shutil.move(filename, f"/content/thesis/data/{filename}")
print(f"Data file ready: /content/thesis/data/{filename}")

# --- Cell 5: Run the pipeline ---
from climate_detector import run_climate_detection

df = run_climate_detection(
    input_path=f"/content/thesis/data/{filename}",
    output_path="/content/thesis/results/detector_results.xlsx",
    batch_size=32,
)

# --- Cell 6: Quick inspection ---
print(df.head(10))
print(f"\nTotal paragraphs: {len(df)}")
print(f"Climate-related distribution:\n{df['detector_label'].value_counts()}")

# --- Cell 7: Download results ---
files.download("/content/thesis/results/detector_results.xlsx")

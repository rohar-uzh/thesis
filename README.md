# Climate Risk Disclosure Quality in European Banks
Master's thesis – University of Zurich, Banking & Finance

## Repository Structure

```
thesis/
├── pdf_parser.py                   # PyMuPDF paragraph extractor
├── climate_detector.py             # ClimateBERT detection pipeline
├── climate_specificity.py          # ClimateBERT specificity classifier
├── climate_commitment.py           # ClimateBERT commitment classifier
├── climate_tcfd.py                 # ClimateBERT TCFD pillar classifier
├── build_dqi.py                    # DQI construction (z-score + raw average)
├── run_regression.py               # Cross-sectional OLS regression (HC3 SEs)
├── run_robustness.py               # Robustness checks and H1 paired t-tests
├── run_tcfd_analysis.py            # TCFD descriptive analysis and correlations
├── requirements.txt
├── .gitignore
│
├── notebooks/                      # Colab notebooks — open directly via badge links below
│   ├── climatebert_detector_colab.ipynb
│   ├── climatebert_specificity_colab.ipynb
│   ├── climatebert_commitment_colab.ipynb
│   ├── climatebert_tcfd_colab.ipynb
│   └── pdfparser.ipynb
│
├── data/                           # Input data (gitignored)
│   ├── bank_data.xlsx              # Sample tracking + bank characteristics
│   └── pdfs/                       # Raw bank reports (BankName_Year_Type.pdf)
│
└── results/                        # All outputs (gitignored)
    ├── parsed/                     # PDF parser output (→ input for detector)
    ├── detector/                   # Climate detection output (→ input for steps 3–5)
    ├── specificity/                # Specificity classifier output
    ├── commitment/                 # Commitment classifier output
    ├── tcfd/                       # TCFD pillar classifier output
    ├── dqi/                        # Disclosure Quality Index output
    └── regression/                 # Regression and robustness results
```

## Reproducibility

All code used in the thesis is available in this repository. The Colab notebooks can be opened directly in Google Colab without any local setup — click the badge at the top of each notebook, or use the links below.

| Notebook | Purpose | Open |
|---|---|---|
| `climatebert_detector_colab.ipynb` | Step 2: Climate paragraph detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohar-uzh/thesis/blob/main/notebooks/climatebert_detector_colab.ipynb) |
| `climatebert_specificity_colab.ipynb` | Step 3: Specificity classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohar-uzh/thesis/blob/main/notebooks/climatebert_specificity_colab.ipynb) |
| `climatebert_commitment_colab.ipynb` | Step 4: Commitment classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohar-uzh/thesis/blob/main/notebooks/climatebert_commitment_colab.ipynb) |
| `climatebert_tcfd_colab.ipynb` | Step 5: TCFD pillar classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohar-uzh/thesis/blob/main/notebooks/climatebert_tcfd_colab.ipynb) |

**To reproduce the full pipeline:**
1. Obtain the 94 annual report PDFs (listed in Appendix A of the thesis) and place them in `data/pdfs/`
2. Run `pdfparser.ipynb` to extract paragraphs → `results/parsed/`
3. Run the four Colab notebooks in the order above (steps 3 and 4 are independent and can run in parallel)
4. Run `build_dqi.py`, `run_regression.py`, `run_robustness.py`, and `run_tcfd_analysis.py` locally

Notebooks are committed with cell outputs stripped. Run each notebook top to bottom in a fresh Colab session to regenerate results.

## Workflow

### Colab steps (GPU required)

1. Edit Python modules locally and `git push` to GitHub via PowerShell
2. Open notebook in Colab — Cell 2 runs `git pull` to fetch the latest modules
3. **Run Cells 1–4 once** to install packages, pull the repo, and load the model onto the GPU
4. **Repeat Cell 5** for each input file: upload → classify → result appended to queue
5. **Run Cell 6 once** to download all results as a single zip archive

> The model loads once in Cell 4 and stays in GPU memory for the entire session.
> `BATCH_SIZE` defaults to 64 (tuned for a T4 GPU); reduce to 32 if you hit OOM errors.

### Local steps (no GPU needed)

6. Save downloaded results to the appropriate `results/` subfolder
7. Run `build_dqi.py` to merge specificity and commitment outputs and construct the DQI
8. Run `run_regression.py` for H2 cross-sectional OLS and extended specifications
9. Run `run_robustness.py` for VIF checks, H1 paired t-tests, and clustered-SE regressions
10. Run `run_tcfd_analysis.py` for TCFD pillar descriptives and correlations with DQI

## PDF Naming Convention

Input PDFs must follow the convention: `BankName_Year_ReportType.pdf`

- Hyphens for multi-word names: `BNP-Paribas_2023_URD.pdf`
- Year must be 4 digits
- Report types: `Annual`, `URD`
- The bank key extracted from the filename must match the `bank` column in `bank_data.xlsx`

## Pipeline Order

```
Step 1  pdf_parser.py           → Parse PDFs into paragraphs         (Colab)
                                  Output: results/parsed/
Step 2  climate_detector.py     → Label climate-related paragraphs   (Colab)
                                  Output: results/detector/
                                          │
                                ┌─────────┴─────────┐
Step 3  climate_specificity.py  →        Step 4  climate_commitment.py
        Specificity classifier           Commitment classifier
        Input: results/detector/         Input: results/detector/
        Output: results/specificity/     Output: results/commitment/
                                └─────────┬─────────┘
Step 5  build_dqi.py            → Merge outputs, construct DQI       (local)
                                  Output: results/dqi/
Step 6  run_regression.py       → Cross-sectional OLS regression     (local)
                                  Output: results/regression/
Step 7  run_robustness.py       → Robustness checks + H1 t-tests     (local)
                                  Output: results/regression/
Step 8  run_tcfd_analysis.py    → TCFD descriptives + correlations   (local)
                                  Output: results/tcfd/
```

> Steps 3 and 4 are **independent and parallel** — both take the detector output
> directly and can be run in any order or simultaneously in separate Colab sessions.

## Models Used

| Model | Purpose |
|---|---|
| `climatebert/distilroberta-base-climate-detector` | Binary climate paragraph detection |
| `climatebert/distilroberta-base-climate-specificity` | Specific vs. vague disclosure |
| `climatebert/distilroberta-base-climate-commitment` | Commitment/action detection |
| `climatebert/distilroberta-base-climate-tcfd` | TCFD pillar classification |

## Disclosure Quality Index (DQI)

```
DQI_i     = mean( Z(Coverage_i), Z(Specificity_i), Z(Commitment_i) )   # main spec
DQI_alt_i = mean( Coverage_i, Specificity_i, Commitment_i )             # robustness
```

Where:
- `Coverage_i`    = climate paragraphs / total paragraphs
- `Specificity_i` = specific climate paragraphs / climate paragraphs
- `Commitment_i`  = commitment climate paragraphs / climate paragraphs

## Regression Specification

```
DQI_i = α + β1·Size_i + β2·ROA_i + β3·CET1_i + ε_i
```

Where `Size = log(Total Assets in EUR bn)`. HC3 heteroskedasticity-robust standard errors throughout.
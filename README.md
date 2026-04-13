# Climate Risk Disclosure Quality in European Banks
Master's thesis – University of Zurich, Banking & Finance

## Structure

```
thesis/
├── pdf_parser.py                   # PyMuPDF paragraph extractor
├── climate_detector.py             # ClimateBERT detection pipeline
├── climate_specificity.py          # ClimateBERT specificity classifier
├── climate_commitment.py           # ClimateBERT commitment classifier
├── build_dqi.py                    # DQI construction (z-score + raw average)
├── run_regression.py               # Cross-sectional OLS regression (HC3 SEs)
├── requirements.txt
├── .gitignore
│
├── notebooks/                      # Colab notebooks (gitignored)
│   ├── climatebert_detector_colab.py
│   ├── climatebert_specificity_colab.ipynb
│   └── climatbert_commitment_colab.ipynb
│
├── data/                           # Input data (gitignored)
│   ├── bank_data.xlsx              # Sample tracking + bank characteristics
│   └── pdfs/                       # Raw bank reports (BankName_Year_Type.pdf)
│
└── results/                        # All outputs (gitignored)
    ├── parsed/                     # PDF parser output
    ├── detector/                   # Climate detection output
    ├── specificity/                # Specificity classifier output
    ├── commitment/                 # Commitment classifier output
    ├── dqi/                        # Disclosure Quality Index output
    └── regression/                 # Regression results
```

## Workflow

1. Edit Python modules locally
2. `git push` to GitHub via PowerShell
3. In Colab: `!cd /content/thesis && git pull`
4. Run Colab notebook cells (GPU inference)
5. Download results via `files.download()` and save to `results/`
6. Run `build_dqi.py` and `run_regression.py` locally (no GPU needed)

## PDF Naming Convention

Input PDFs must follow the convention: `BankName_Year_ReportType.pdf`
- Hyphens for multi-word names: `BNP-Paribas_2023_Annual.pdf`
- Report types: `Annual`, `Sustainability`, `Pillar3`
- The bank key extracted from the filename must match the `bank` column in `bank_data.xlsx`

## Models Used

| Model | Purpose |
|---|---|
| `climatebert/distilroberta-base-climate-detector` | Binary climate paragraph detection |
| `climatebert/distilroberta-base-climate-specificity` | Specific vs. vague disclosure |
| `climatebert/distilroberta-base-climate-commitment` | Commitment/action detection |

## Disclosure Quality Index (DQI)

```
DQI_i     = mean( Z(Coverage_i), Z(Specificity_i), Z(Commitment_i) )   # main spec
DQI_alt_i = mean( Coverage_i, Specificity_i, Commitment_i )             # robustness
```

## Regression Specification

```
DQI_i = α + β1·Size_i + β2·ROA_i + β3·CET1_i + ε_i
```

Where `Size = log(Total Assets in EUR bn)`. HC3 heteroskedasticity-robust standard errors throughout.

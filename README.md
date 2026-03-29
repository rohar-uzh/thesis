# Climate Risk Disclosure Quality in European Banks

Master's thesis – University of Zurich, Banking & Finance

## Structure

```
thesis/
├── climate_detector.py      # ClimateBERT detection pipeline
├── notebooks/
│   └── colab_detector.py    # Colab orchestration (paste into notebook)
├── data/                    # Your input Excel files (gitignored)
├── results/                 # Model outputs (gitignored)
├── requirements.txt
└── .gitignore
```

## Workflow

1. Edit Python modules locally (or via Claude)
2. `git push` to GitHub
3. In Colab: `!cd /content/thesis && git pull`
4. Run the notebook cells

## Models Used

- `climatebert/distilroberta-base-climate-detector` — binary climate paragraph detection
- (Planned) `climatebert/distilroberta-base-climate-specificity`
- (Planned) `climatebert/distilroberta-base-climate-commitment`

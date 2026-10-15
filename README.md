# Project 1 — Customer Segmentation using K-Means

## Run
From this folder:
```bash
python customer_segmentation.py
```

## What it does
- Cleans missing/duplicate rows and checks required columns.
- Standardizes age, annual income, spending score, and monthly purchase frequency.
- Tests K values from 2 to 8 (limited by dataset size) and selects the best silhouette score.
- Fits K-Means and Agglomerative (Ward) clustering for comparison.
- Saves cluster assignments, customer profiles, evaluation metrics, and visualizations to `outputs/`.

`data/customer_data.csv` is a synthetic sample dataset for a repeatable demonstration, not a real customer database.

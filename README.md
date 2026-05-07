# RACE Reading Comprehension and Quiz Generation System

Course project for **AL2002 Artificial Intelligence (Lab) — Spring 2026**.
An end-to-end classical-ML pipeline that ingests reading passages, verifies multiple-choice answers, generates plausible distractors, and produces graduated hints — all wrapped in a 4-screen Streamlit app.

> Spec reference: `AL2002_LabProject.pdf` shipped alongside this repo. This implementation follows the **Traditional ML** track end-to-end (no neural networks).

---

## 1. Architecture Overview

```
[RACE CSVs] -> preprocessing -> One-Hot features (+ optional TF-IDF) -> Models -> Streamlit UI
```

- **Model A — Q&A Verifier + Question Generator**
  - Logistic Regression and Linear SVM verifiers
  - Multinomial Naive Bayes question-type classifier
  - K-Means + Gaussian Mixture clustering (unsupervised)
  - Soft-voting ensemble (LR + SVM + NB)
  - Wh-template question generator with a Random Forest ranker
- **Model B — Distractor + Hint Generator**
  - Distractor candidate extraction + Logistic Regression ranker (with diversity penalty)
  - Graduated hint ranker (LR over sentence features) with Hint 1/2/3 thresholds
- **UI** — Streamlit multi-page app with sidebar navigation, caching, session log, CSV export.

The **primary** classical text representation is **One-Hot Encoding** (binary bag-of-words via `CountVectorizer(binary=True)`), per the project spec. **TF-IDF** is included as an *optional* secondary representation for ablation.

---

## 2. Folder Layout

```
.
├── data/
│   ├── raw/                  # train.csv, val.csv, test.csv (you place these here)
│   └── processed/            # cached splits and feature matrices
├── models/
│   ├── model_a/traditional/  # LR, SVM, NB, K-Means, GMM, ensemble, vectorizer
│   └── model_b/traditional/  # distractor ranker, hint ranker, vectorizers
├── src/
│   ├── preprocessing.py      # cleaning, normalization, Kaggle format support
│   ├── features.py           # One-Hot vectorizer + handcrafted lexical features (+ optional TF-IDF)
│   ├── model_a_train.py      # Model A: verifier, NB, clustering, ensemble
│   ├── model_b_train.py      # Model B: distractor + hint rankers
│   ├── question_generation.py# Wh-template question generator + RF ranker
│   ├── inference.py          # unified runtime API used by the UI
│   ├── evaluate.py           # all metrics + writes metrics.json
│   ├── train_all.py          # one-command reproducible pipeline
│   └── demo_data.py          # tiny synthetic RACE-like fallback for the UI
├── ui/
│   ├── app.py                # Streamlit entrypoint (4 screens)
│   └── components/           # reusable widgets
├── notebooks/
│   └── EDA.ipynb             # exploratory data analysis (9 sections with charts)
├── tests/
│   └── test_inference.py     # smoke tests on demo_data
├── report/
│   └── final_report.md       # completed final report with results and analysis
├── requirements.txt
└── README.md
```

---

## 3. Quickstart

### 3.1 Set up the environment (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 Run the UI immediately on synthetic demo data (no Kaggle download needed)

```powershell
streamlit run ui\app.py
```

The app auto-falls-back to `src/demo_data.py` if no trained models or RACE CSVs are present, so all 4 screens are usable from minute one.

### 3.3 Train on the real RACE dataset

1. Download the RACE CSV splits (search Kaggle for **"RACE Reading Comprehension Dataset"**) and place them in:

   ```
   data/raw/train.csv
   data/raw/val.csv
   data/raw/test.csv
   ```

   Supported formats:
   - **Expanded format:** `id, article, question, A, B, C, D, answer`
   - **Kaggle format:** `example_id, article, question, options, answer` (auto-detected and normalized)

2. Train every model in one command (subsample for laptops):

   ```powershell
   python -m src.train_all --train_path data\raw\train.csv --val_path data\raw\val.csv --test_path data\raw\test.csv --max_train_rows 20000 --max_val_rows 5000
   ```

   Or full training:

   ```powershell
   python -m src.train_all --train_path data\raw\train.csv --val_path data\raw\val.csv --test_path data\raw\test.csv
   ```

   Optional ablation switching feature representation to TF-IDF:

   ```powershell
   python -m src.train_all --train_path data\raw\train.csv --val_path data\raw\val.csv --test_path data\raw\test.csv --use_tfidf
   ```

3. Launch the UI again — it will automatically pick up the trained artifacts in `models/`.

   ```powershell
   streamlit run ui\app.py
   ```

### 3.4 Run tests

```powershell
python -m unittest tests.test_inference -v
```

---

## 4. Reproducibility

- Every training entrypoint uses `random_state=42`.
- `src/train_all.py` is the single shell command required by spec section 10.1.
- Each training run writes `models/manifest.json` recording timestamp, sklearn version, training row count, and the metrics achieved.

## 5. Key Results

| Component          | Accuracy | Macro-F1 |
|--------------------|----------|----------|
| Logistic Regression | 0.969    | 0.956    |
| SVM                | 0.969    | 0.956    |
| Ensemble           | 0.969    | 0.956    |
| NB Question-Type   | 1.000    | 1.000    |

| Sub-task          | Precision | Recall | F1    |
|-------------------|-----------|--------|-------|
| Distractor Ranker | 0.759     | 0.917  | 0.830 |

Full results and analysis: see `report/final_report.md`.

## 6. Evaluation

Metrics are computed by `src/evaluate.py` and persisted to `models/metrics.json`:

- **Model A**: Accuracy, Macro F1, Exact Match (verifier); Silhouette and Purity (clustering).
- **Model B**: Precision / Recall / F1 (distractors), Top-1 distractor accuracy, Precision@K and R² (hint scorer).
- **Confusion matrix** plotted live in the Analytics screen.

## 7. Notes & Limitations

- "One-Hot Encoding" is implemented as `CountVectorizer(binary=True)` over a 10k-term vocabulary (presence/absence of each vocab term per document) — the standard sklearn realization for variable-length text.
- All inference completes in well under the 10-second-per-request constraint from spec section 10.1.
- The dataset is sourced from Chinese school examinations; refer to the report's *Ethical Considerations* section for the bias discussion required by spec 10.2.

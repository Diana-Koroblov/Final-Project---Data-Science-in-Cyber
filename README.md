# Reproduction & Critical Evaluation — ACR'25

**Course:** Data Science in Cybersecurity  
**Instructor:** Dr. Uri Itai  
**Submission deadline:** Friday, July 10, 2026

---

## Project Description

This project critically evaluates and reproduces the paper:

> **"Performance of Machine Learning Classifiers for Anomaly Detection in Cyber Security Applications"**  
> Markus Haug & Gissel Velarde — IU International University of Applied Sciences  
> *Proceedings of the Third International Conference on Advances in Computing Research (ACR'25), Springer Nature, 2025, pp. 285–294*

The paper benchmarks machine learning classifiers (XGBoost, MLP, GAN, VAE, MO-GAAL) on two imbalanced cybersecurity datasets. This reproduction study:

- Reproduces the core XGBoost results under a corrected, leak-free pipeline
- Critically evaluates the authors' methodology and claims
- Extends the evaluation with Logistic Regression and Random Forest baselines
- Adds MCC, ROC-AUC, Fβ, confusion matrices, error analysis, and threshold optimisation — metrics absent from the original paper

Key finding: the paper's *written* methodology implies data leakage (encode → scale → split), but the *actual code* wraps preprocessing in a scikit-learn Pipeline and avoids it. The text/code discrepancy is itself a reproducibility failure.

---

## Links

| Resource | URL |
|---|---|
| Original paper (Springer) | https://doi.org/10.1007/978-3-031-87647-9_25 |
| Original GitHub repository | https://github.com/markushaug/acr-25 |
| KDDCUP99 dataset | https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html |
| Credit Card Fraud dataset | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |

---

## Repository Contents

```
├── notebook.ipynb          # Main reproduction notebook (self-contained, executable)
├── report.docx             # PDF report (critical evaluation + reproducibility analysis)
├── anomaly.pdf             # Original paper (ACR'25)
├── acr-25/                 # Original authors' code (cloned from markushaug/acr-25)
│   ├── models/             # Authors' Jupyter notebooks
│   ├── requirements.txt    # Authors' dependencies (note: contains broken local path)
│   └── README.md           # Authors' setup instructions
└── data/                   # Dataset files (not tracked by git — see Dataset Setup below)
```

---

## Dataset Setup

The datasets are **not included** in this repository due to size. Download them before running the notebook.

### KDDCUP99 (`corrected.gz`)

The paper uses the **corrected** test set (311,029 rows), not the commonly downloaded 10% training sample. The sklearn loader fetches the correct file automatically:

```python
from sklearn.datasets import fetch_kddcup99
data = fetch_kddcup99(subset='SA', shuffle=False, random_state=42)
```

Alternatively, download `corrected.gz` directly from the UCI repository and place it under `data/kdd/`.

### Credit Card Fraud (2013)

Download `creditcard.csv` from Kaggle:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it at `data/creditcard/creditcard.csv`.

---

## Execution Instructions

### Requirements

- Python 3.9 or later
- Jupyter Notebook or JupyterLab

### Setup

```bash
# 1. Clone this repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
scipy
jupyter
```

### Run the notebook

```bash
jupyter notebook notebook.ipynb
```

Run all cells top-to-bottom (**Kernel → Restart & Run All**). Expected runtime: ~5–10 minutes depending on hardware. All random seeds are fixed (`RANDOM_STATE = 42`) for full reproducibility.

---

## Key Results

| Model | Dataset | F1 | AUC |
|---|---|---|---|
| Logistic Regression | KDDCUP99 | 0.975 | 0.994 |
| Random Forest | KDDCUP99 | 0.985 | 0.998 |
| XGBoost | KDDCUP99 | **0.988** | **0.998** |
| Logistic Regression | Credit Card Fraud | 0.114 | 0.972 |
| Random Forest | Credit Card Fraud | 0.841 | 0.957 |
| XGBoost | Credit Card Fraud | **0.858** | **0.968** |

XGBoost's superiority is confirmed under a controlled, fair comparison (no hyperparameter tuning advantage). For Credit Card Fraud, the LR result (low Precision, high AUC) is a threshold artefact, not model failure — threshold optimisation at 0.34 improves XGBoost Recall to 0.857 at Precision 0.866.

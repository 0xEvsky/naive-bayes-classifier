# Naive Bayes Classifier & PCA from Scratch

A machine learning assignment implementing Naive Bayes classification and Principal Component Analysis (PCA) from scratch using Python and NumPy. The project covers two datasets — one categorical and one numerical — and compares three experimental setups: baseline, feature selection, and PCA-based dimensionality reduction.

---

## Project Structure

```
├── cate_nb.ipynb        # Categorical Naive Bayes — Mushroom dataset
├── minist.ipynb         # Gaussian Naive Bayes — MNIST dataset
├── data_preprocess.py   # PCA, FeatureSelection, and encoding utilities
├── data/                # Datasets (not tracked by git)
└── README.md
```

---

## Datasets

- **Mushroom** (categorical) — UCI Machine Learning Repository. 8,124 samples, 22 features, binary classification (edible / poisonous).
- **MNIST** (numerical) — Loaded via `sklearn.datasets.fetch_openml`. 70,000 samples, 784 pixel features, 10-class digit classification.

---

## How to Run

**1. Install dependencies**

```bash
pip install numpy pandas scikit-learn
```

**2. Download the Mushroom dataset**

Place `mushrooms.csv` inside a `data/` folder in the project root.
You can download it from: https://www.kaggle.com/datasets/uciml/mushroom-classification

**3. Run the notebooks**

Open and run all cells in order:

```bash
jupyter notebook cate_nb.ipynb    # Mushroom experiments
jupyter notebook minist.ipynb     # MNIST experiments
```

MNIST data is fetched automatically via `fetch_openml` on first run (requires internet connection).

---

## Results Summary

| Dataset  | Baseline | Feature Selection | PCA       |
|----------|----------|-------------------|-----------|
| Mushroom | 99.57%   | —                 | —         |
| MNIST    | 72.79%   | 73.11%            | **87.20%**|
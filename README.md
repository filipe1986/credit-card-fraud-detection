# Credit Card Fraud Detection (IBM Snap ML vs. Scikit-Learn)

## Overview
This project focuses on building and evaluating machine learning models to identify fraudulent credit card transactions. Utilizing a dataset of 284,807 transactions, the analysis addresses severe class imbalance and compares the computational efficiency and performance of **IBM Snap ML** against standard **Scikit-Learn** implementations.

## Key Financial Insights & Data Discovery
* **Extreme Class Imbalance:** Fraudulent transactions represent a tiny fraction of total activity, requiring class-weighted loss functions (`sample_weight='balanced'`) during model training.
* **Transaction Skewness:** 90% of transaction amounts are $203.00 or lower, yet the maximum reaches $25,691.16. Logarithmic scaling was applied to visualize the long-tail distribution effectively.
* **Feature Anonymization:** Predictors $V_1$ through $V_{28}$ represent PCA-transformed features to ensure consumer data privacy.

## Modeling & Performance Benchmarks
Two primary classifiers were trained and evaluated on an $L_1$-normalized feature space:

1. **Decision Tree Classifier (`max_depth=4`)**
   * Benchmarked training times between Scikit-Learn and Snap ML on scaled transaction data.
2. **Support Vector Machine (SVM)**
   * **Scikit-Learn Hinge Loss:** ~0.23367
   * **Snap ML Hinge Loss:** ~0.22818
   * *Conclusion:* Both models achieved comparable decision boundaries, with Snap ML offering faster execution times suitable for large-scale financial data pipelines.

## Repository Structure
```text
├── credit_card_fraud_detection.ipynb  # Main Jupyter analysis notebook
├── .gitignore                         # Ignores venv, bytecode, and CSV data
└── README.md                          # Executive project summary


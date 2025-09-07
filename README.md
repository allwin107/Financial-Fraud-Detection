# Financial Fraud Detection Model

## Overview
This Jupyter notebook implements a machine learning solution for detecting fraudulent financial transactions. It addresses class imbalance, performs feature engineering, trains multiple models, and evaluates performance using metrics like F1-score and ROC-AUC. Synthetic data is used for demonstration due to the large original dataset (6M+ rows).

## Problem Statement
Financial fraud causes billions in losses. This model uses ML to detect fraud in real-time mobile money transactions, minimizing false positives/negatives.

## Dataset
- **Original**: 6,362,620 transactions × 10 features (e.g., amount, type, balances, isFraud).
- **Synthetic**: 100,000 rows generated for demo, mimicking real patterns (0.1% fraud rate).
- Source: Simulated mobile money data; real dataset not included (load via `load_and_explore_data` function).

## Requirements
- Python 3.12+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn, xgboost, lightgbm.
Install via:
```
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost lightgbm
```

## Installation
1. Clone repo: `git clone <repo-url>`.
2. Install dependencies: `pip install -r requirements.txt` (create if needed).
3. Run in Jupyter: `jupyter notebook Fraud_Transaction_Prediction_for_Financial_Companies.ipynb`.

## Usage
1. Update dataset path in Section 2 if using real data.
2. Execute cells sequentially.
3. Outputs: Data exploration, model evaluations, visualizations, business insights.
4. For production: Use `create_prediction_pipeline` for inference.

## Notebook Structure
1. **Installation Requirements**: Pip installs.
2. **Environment Setup**: Imports libraries.
3. **Data Loading**: Loads/synthesizes data.
4. **Data Quality Assessment**: Checks missing/duplicates.
5. **Exploratory Data Analysis**: Stats, distributions, correlations.
6. **Feature Engineering**: Creates new features (e.g., ratios, temporal).
7. **Data Preprocessing**: Encoding, scaling, splitting.
8. **Handling Class Imbalance**: SMOTE, undersampling.
9. **Model Training & Tuning**: Logistic Regression, Random Forest, etc., with GridSearchCV.
10. **Model Evaluation**: Metrics, ROC/PR curves, confusion matrices.
11. **Feature Importance**: Analyzes top predictors.
12. **Business Insights**: Fraud patterns, risk factors.
13. **Utilities**: Prediction pipeline, monitoring.

## Key Results
- Best model (e.g., XGBoost): High recall (~0.9+), F1 ~0.8+ on synthetic data.
- Top features: oldbalanceOrg, amount, type_TRANSFER.
- Fraud patterns: High in TRANSFER/CASH_OUT, large amounts, night-time.

## Limitations
- Synthetic data may not capture all real patterns.
- No real-time deployment code.
- Assumes balanced hardware for large datasets.

## Future Work
- Integrate real dataset.
- Add deep learning models.
- Deploy as API.

## License
MIT License. See LICENSE file.

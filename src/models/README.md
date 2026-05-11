# XGBoost Phishing Detection Model

Train and deploy XGBoost classifier for SSL certificate-based phishing detection.

## Overview

This module trains an XGBoost binary classifier to detect phishing domains based on SSL certificate and domain features. The model uses the top N features (by effect size) identified through statistical analysis.

## Quick Start

### 1. Prerequisites

First, run the EDA notebook to generate feature statistics:

```bash
jupyter notebook notebooks/eda_sandbox.ipynb
# Run all cells to generate: sources/processed/comprehensive_feature_stats.csv
```

### 2. Train Model

Train XGBoost model using top 10 features:

```bash
python src/models/train_xgboost.py --top-n 10
```

**Output:**
- `models/xgboost_phishing_detector.joblib` - Trained model
- `models/feature_list.json` - Features used
- `models/metrics.json` - Performance metrics
- `models/*.png` - Evaluation plots

### 3. Make Predictions

**Batch prediction on cert file:**

```bash
python src/models/predict.py \
    --input sources/raw/certs_new.jsonl \
    --output predictions.csv
```

**Single domain check:**

```bash
python src/models/predict.py --domain "paypa1-secure.com"
```

## Training Pipeline

### Step 1: Feature Extraction

The training script automatically:
1. Loads live certificates (`sources/raw/certs_live.jsonl`)
2. Loads historical phishing certificates (`sources/raw/certs_phishing_historical.jsonl`)
3. Extracts comprehensive features using `FeatureEngineer`
4. Merges datasets (live = legitimate, historical = phishing)

### Step 2: Feature Selection

Reads `comprehensive_feature_stats.csv` and selects top N features by Cohen's d (effect size).

Example top features:
- `brand_distance` - Levenshtein distance to known brands
- `keyword_count` - Number of phishing keywords
- `validity_days` - Certificate validity period
- `entropy` - Domain entropy (randomness)
- `tld_risk` - TLD risk score

### Step 3: Model Training

**XGBoost Hyperparameters:**
- `max_depth`: 6
- `learning_rate`: 0.1
- `n_estimators`: 200
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `scale_pos_weight`: Auto-calculated to handle class imbalance

**Train-test split:** 80/20 stratified

### Step 4: Evaluation

Generates comprehensive evaluation:
- Classification report (precision, recall, F1)
- Confusion matrix
- ROC curve (AUC score)
- Precision-Recall curve
- Feature importance plot

## Command-Line Options

### train_xgboost.py

```bash
python src/models/train_xgboost.py [OPTIONS]

Options:
  --live-certs PATH         Live cert JSONL file (default: sources/raw/certs_live.jsonl)
  --hist-certs PATH         Historical phishing JSONL (default: sources/raw/certs_phishing_historical.jsonl)
  --feature-stats PATH      Feature statistics CSV (default: sources/processed/comprehensive_feature_stats.csv)
  --top-n INT               Number of top features to use (default: 10)
  --test-size FLOAT         Test set proportion (default: 0.2)
  --output-dir PATH         Output directory (default: models)
```

### predict.py

```bash
python src/models/predict.py [OPTIONS]

Options:
  --model-dir PATH          Model directory (default: models)
  --input PATH              Input JSONL file (for batch prediction)
  --output PATH             Output CSV file (for batch prediction)
  --domain STRING           Single domain to check
  --threshold FLOAT         Classification threshold (default: 0.5)
```

## Model Performance

Expected performance on test set (with top 10 features):

| Metric | Value | Description |
|--------|-------|-------------|
| **Precision** | 0.85-0.95 | Of predicted phishing, % that are correct |
| **Recall** | 0.80-0.90 | Of actual phishing, % that are detected |
| **F1 Score** | 0.82-0.92 | Harmonic mean of precision/recall |
| **ROC-AUC** | 0.90-0.98 | Overall discriminative ability |

**Note:** Actual performance depends on data quality and class balance.

## Handling Class Imbalance

Phishing domains are rare in live CT logs. The model handles this via:

1. **scale_pos_weight**: Automatically calculated as `n_negative / n_positive`
2. **Stratified split**: Train-test split preserves class proportions
3. **Evaluation metrics**: Focus on precision, recall, F1 (not just accuracy)

## Feature Importance

After training, check `models/feature_importance.png` to see which features the model relies on most.

Example interpretation:
- High importance on `brand_distance` → Model detects typosquatting
- High importance on `validity_days` → Short-lived certs are suspicious
- Low importance on `domain_length` → Length alone is not discriminative

## Model Persistence

The trained model is saved using `joblib` and can be loaded:

```python
import joblib

model = joblib.load('models/xgboost_phishing_detector.joblib')
y_pred_proba = model.predict_proba(X)[:, 1]
```

## Integration with Live Ingestion

To detect phishing in real-time:

1. **Ingest live certificates:**
   ```bash
   python src/data/ingest_certificates_labels.py live-certs --duration 300
   ```

2. **Run predictions:**
   ```bash
   python src/models/predict.py \
       --input sources/raw/certs_live.jsonl \
       --output predictions.csv \
       --threshold 0.7
   ```

3. **Filter high-risk domains:**
   ```bash
   # Get domains with >70% phishing probability
   cat predictions.csv | awk -F',' '$2 > 0.7 {print $1, $2}'
   ```

## Hyperparameter Tuning

To improve performance, tune hyperparameters using grid search:

```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(scale_pos_weight=scale_pos_weight),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

## Troubleshooting

### Error: "Feature statistics file not found"

**Solution:** Run EDA notebook first to generate `comprehensive_feature_stats.csv`:
```bash
jupyter notebook notebooks/eda_sandbox.ipynb
```

### Error: "Live cert file not found"

**Solution:** Collect live data first:
```bash
python src/data/ingest_certificates_labels.py live-certs --duration 300
```

### Error: "Historical cert file not found"

**Solution:** Collect historical phishing data:
```bash
# Step 1: Get phishing labels
python src/data/ingest_certificates_labels.py historical-phishing-labels

# Step 2: Fetch historical certs
python src/data/ingest_certificates_labels.py historical-phishing-certs \
    --labels sources/raw/phishing_labels.jsonl
```

### Warning: Low precision or recall

**Possible causes:**
- Insufficient training data (collect more historical phishing samples)
- Class imbalance too severe (adjust `scale_pos_weight`)
- Poor feature selection (try different top-N values)

**Solutions:**
1. Collect more historical phishing data
2. Try different `--top-n` values (e.g., 15, 20)
3. Check feature importance and remove weak features

## Next Steps

1. **Experiment with feature sets:** Try `--top-n 15` or `--top-n 20`
2. **Tune hyperparameters:** Use grid search for optimal performance
3. **Deploy in production:** Integrate with live ingestion pipeline
4. **Monitor drift:** Retrain periodically as phishing tactics evolve
5. **Ensemble models:** Combine XGBoost with Random Forest or Neural Networks

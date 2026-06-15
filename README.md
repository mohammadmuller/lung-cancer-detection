# Lung Cancer Risk Predictor

A machine learning desktop application for lung cancer risk assessment. The system trains four classical ML models on a tabular clinical dataset and provides a graphical interface for risk prediction, model comparison, and per-patient analysis with personalized recommendations.

---

## Overview

The application accepts 23 clinical and lifestyle features as input — including smoking history, air pollution exposure, genetic risk, respiratory symptoms, and demographic data — and outputs a probabilistic risk score with a five-zone risk classification. Four models are trained simultaneously, allowing the user to compare predictions across algorithms and select the most appropriate model for a given use case.

---

## Features

- **Four-model ensemble** — Random Forest, Logistic Regression, XGBoost, and LightGBM are trained and evaluated in a background thread, keeping the UI responsive during training
- **Class imbalance handling** — automatic `scale_pos_weight` computation for XGBoost and `class_weight='balanced'` for the other models
- **Optimal threshold tuning** — each model uses the decision threshold best suited to imbalanced medical data rather than a fixed default
- **Risk zone chart** — a horizontal bar chart with five colour-coded zones (Very Low / Low / Moderate / High / Very High) rendered via Matplotlib embedded in the UI
- **Per-patient risk factor breakdown** — the report highlights which input features are contributing to the predicted risk level and at what severity
- **Personalized recommendations** — smoking cessation advice, dietary guidance, air pollution precautions, and low-dose CT scan referral are generated based on the specific input values
- **Prediction history** — all predictions within a session are logged with timestamps and key inputs, and can be cleared at any time
- **CSV dataset loader** — any CSV file with the expected column schema can be loaded at runtime; the app handles missing values, unnamed columns, and type coercion automatically

---

## Models and Configuration

| Model | Key Parameters |
|---|---|
| Random Forest | 200 estimators, max depth 12, balanced class weights, all CPU cores |
| Logistic Regression | C=0.1, liblinear solver, balanced class weights, 5000 max iterations |
| XGBoost | 150 estimators, max depth 5, LR 0.1, auto scale_pos_weight |
| LightGBM | 150 estimators, max depth 6, LR 0.1, balanced class weights |

All models are trained on an 80/20 stratified train/test split with `StandardScaler` normalization applied to the feature matrix.

---

## Input Features

The application exposes 23 input features, each scored on a 0-10 scale (except Age and Gender):

| Feature | Range | Description |
|---|---|---|
| Age | 0-120 | Patient age in years |
| Gender | Male / Female | Categorical dropdown |
| Air Pollution | 0-10 | Exposure to air pollution |
| Alcohol Use | 0-10 | Alcohol consumption level |
| Dust Allergy | 0-10 | Severity of dust allergy |
| Occupational Hazards | 0-10 | Workplace hazard exposure |
| Genetic Risk | 0-10 | Hereditary predisposition |
| Chronic Lung Disease | 0-10 | Severity of existing lung disease |
| Balanced Diet | 0-10 | Diet quality (higher = healthier) |
| Obesity | 0-10 | Obesity level |
| Smoking | 0-10 | Smoking intensity |
| Passive Smoker | 0-10 | Secondhand smoke exposure |
| Chest Pain | 0-10 | Frequency and severity |
| Coughing of Blood | 0-10 | Frequency |
| Fatigue | 0-10 | Chronic fatigue level |
| Weight Loss | 0-10 | Unexplained weight loss |
| Shortness of Breath | 0-10 | Breathing difficulty |
| Wheezing | 0-10 | Frequency |
| Swallowing Difficulty | 0-10 | Difficulty swallowing |
| Clubbing of Finger Nails | 0-10 | Nail deformity severity |
| Frequent Cold | 0-10 | Frequency of common cold |
| Dry Cough | 0-10 | Frequency |
| Snoring | 0-10 | Frequency during sleep |

---

## Dataset

The application expects a CSV file with the following required column:

- `Result` — binary target column (0 = no cancer, 1 = cancer)

Optional columns that are automatically dropped if present:

- `Patient Id`
- `Level` (severity label, not used for training)
- Any unnamed or fully empty columns

All other numeric columns are treated as features. Missing values are imputed with the column median.

A compatible dataset is publicly available on [Kaggle](https://www.kaggle.com) — search for **Lung Cancer dataset** with the matching feature schema.

---

## Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/your-username/lung-cancer-risk-predictor.git
cd lung-cancer-risk-predictor
pip install -r requirements.txt
```

**Dependencies:**

```
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
PyQt5
```

---

## Usage

```bash
python lung_cancer_prediction.py
```

The application opens directly to the prediction interface. Before making predictions, go to the **Model Training** tab to load a dataset and train the models.

### Workflow

1. Open the **Model Training** tab
2. Click **Load Dataset** and select a compatible CSV file
3. Click **Start Training** — progress is shown in a progress bar; the UI remains responsive
4. Once training is complete, switch to the **Risk Prediction** tab
5. Enter patient data using the input fields and spinboxes
6. Select a model from the dropdown and click **Predict Cancer Risk**
7. The result panel shows the risk percentage, zone classification, a bar chart, and a detailed per-factor analysis
8. All predictions are logged automatically in the **Prediction History** tab

---

## Interface Tabs

| Tab | Description |
|---|---|
| Risk Prediction | Input panel with 23 features, model selector, risk bar chart, and detailed report |
| Model Training | Dataset loader, training progress bar, dataset statistics summary |
| Prediction History | Scrollable log of all predictions in the current session with timestamps |

---

## Risk Zone Classification

| Zone | Range | Color |
|---|---|---|
| Very Low | 0 - 20% | Green |
| Low | 20 - 35% | Light Green |
| Moderate | 35 - 55% | Amber |
| High | 55 - 75% | Orange |
| Very High | 75 - 100% | Red |

---

## Notes

- This tool is intended as a clinical decision-support aid and does not replace professional medical diagnosis. All outputs should be reviewed by a qualified healthcare provider.
- Models are not persisted between sessions; retraining is required each time the application is launched.
- The background training thread (`TrainingThread`) emits progress signals to the main UI thread — do not close the application during training.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

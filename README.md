# Predicting Student Mental Health Score from Social Media Usage

A machine learning regression project that predicts a student's `Mental_Health_Score` (continuous, roughly 3 to 10) from their social media habits, study time, sleep, physical activity and stress level.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Using the Saved Model](#using-the-saved-model)
- [Notes and Limitations](#notes-and-limitations)
- [Next Steps](#next-steps)

## Overview
This is a **regression** problem: the target is a number, not a category. The notebook walks through the full ML workflow, from exploration and cleaning to a tuned model saved as a single reusable pipeline:

1. Exploratory data analysis (EDA)
2. Data cleaning
3. Feature engineering and encoding
4. A scikit-learn `Pipeline` with a `ColumnTransformer`
5. Model comparison (Linear Regression vs. Random Forest) and hyperparameter tuning
6. Saving the pipeline with `joblib`

## Dataset
**File:** `Student Social Media And Mental Health Impact.csv`, with 5,000 students and 13 columns.

| Group | Columns |
|-------|---------|
| Demographics | `Age`, `Gender`, `Country`, `Academic_Level` |
| Platform usage | `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Most_Used_Platform`, `Purpose_Of_Use` |
| Lifestyle | `Study_Hours`, `Sleep_Hours_Per_Night`, `Physical_Activity_Hours` |
| Stress | `Stress_Level` (Low / Medium / High / Very High) |
| **Target** | `Mental_Health_Score` |

## Project Workflow

### 1. EDA
Plots in the notebook:
- Distribution of the target
- Correlation heatmap
- Stress level vs. score (boxplot)
- Daily usage hours vs. score
- Sleep hours vs. score
- Most-used platform counts (Instagram, TikTok and Facebook are the top three)

### 2. Data cleaning
- Dropped duplicate rows (2 found).
- `Physical_Activity_Hours` had impossible negative values (minimum -0.4), so they were clipped to 0 instead of dropping those rows.
- No missing values in the dataset.
- IQR check flagged a few outliers: 2 in `Study_Hours` and 22 in `Physical_Activity_Hours`.

### 3. Skewness check
Most numeric columns were close to symmetric. `Study_Hours` was the most skewed (about 0.44), so it gets a `log1p` transform.

### 4. Feature engineering
`Country` has 111 unique values, which would create too many one-hot columns. The top 10 countries are kept and all others are grouped as `Other`, giving 11 categories in a new `Grouped_country` column.

### 5. Preprocessing (`ColumnTransformer`)
| Feature type | Columns | Treatment |
|--------------|---------|-----------|
| Skewed numeric | `Study_Hours` | `log1p` then `StandardScaler` |
| Other numeric | `Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night` | `StandardScaler` |
| Ordinal | `Stress_Level` | `OrdinalEncoder` (Low < Medium < High < Very High) |
| Nominal | `Gender`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, `Grouped_country` | `OneHotEncoder(handle_unknown="ignore")` |

### 6. Modeling
- **Train/test split:** 70/30, `random_state=42`, done before fitting any preprocessing to avoid data leakage.
- **Models:** Linear Regression (baseline), Random Forest (default), Random Forest tuned with `RandomizedSearchCV` (15 iterations, 5-fold CV, scored on R²).
- **Best parameters found:** `n_estimators=200`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2`.

## Results
Evaluated on the held-out test set:

| Model | Test R² | Train R² | MAE | RMSE |
|-------|--------:|---------:|----:|-----:|
| Linear Regression | 0.740 | 0.724 | 0.536 | 0.676 |
| Random Forest (default) | **0.878** | 0.981 | **0.347** | **0.464** |
| Random Forest (tuned) | 0.865 | 0.955 | 0.369 | 0.487 |

**Takeaways**
- Random Forest clearly beats the linear baseline, so there are non-linear patterns in the data.
- The default Random Forest overfits (train R² 0.98 vs. test 0.88). Tuning reduced the gap but did not improve test scores on this split.
- On average, predictions are off by about 0.35 to 0.37 score points.

## Tech Stack
Python 3, pandas, NumPy, matplotlib, seaborn, scikit-learn, joblib. The notebook was run in Google Colab (Python 3.12).

## Getting Started
```bash
git clone <your-repo-url>
cd <your-repo>
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```
1. Place `Student Social Media And Mental Health Impact.csv` in the project folder.
2. Open `ML_Project.ipynb` and run all cells.

## Using the Saved Model
The notebook saves the **entire pipeline** (preprocessing plus model) as `Mental_Health_Model.pkl`, so you can predict on raw input:

```python
import joblib
import pandas as pd

model = joblib.load("Mental_Health_Model.pkl")

sample = pd.DataFrame([{
    "Study_Hours": 3.0,
    "Age": 21,
    "Avg_Daily_Usage_Hours": 5.0,
    "Daily_Unlocks": 170,
    "Physical_Activity_Hours": 1.0,
    "Sleep_Hours_Per_Night": 7.0,
    "Stress_Level": "Medium",
    "Gender": "Female",
    "Academic_Level": "Undergraduate",
    "Most_Used_Platform": "Instagram",
    "Purpose_Of_Use": "Entertainment",
    "Grouped_country": "India",
}])
print(model.predict(sample))
```
Use the same scikit-learn version the model was trained with, since pickles are not guaranteed to load across versions.

## Notes and Limitations
- The saved `Mental_Health_Model.pkl` is the **default** Random Forest (`rf_pipeline`), not the tuned one (`rf_best_pipeline`). Save the tuned pipeline instead if you prefer it.
- The data looks survey-based or synthetic. The model finds correlations, not causes, and should not be used for any clinical or diagnostic purpose.
- Results come from a single train/test split. The Random Forest's gap between training and test scores suggests some overfitting.

## Next Steps
- Build a FastAPI backend with Pydantic validation and a `/predict` endpoint that loads the saved pipeline.
- Connect a simple frontend and deploy it as a live demo.
- Add feature-importance analysis and try further models (such as gradient boosting).

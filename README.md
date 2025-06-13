# 🏡 California Housing Price Prediction using Linear, Lasso, and Random Forest Regression

This project predicts median house values in California using multiple regression techniques, emphasizing the complete ML workflow — from raw data to model deployment. It includes comprehensive data preprocessing, exploratory analysis, feature engineering, model training, evaluation, and visualization.

---

## 📌 Overview

I used the **California Housing Prices dataset** to build and evaluate three machine learning models:

- **Linear Regression**
- **Lasso Regression** (with alpha tuning via cross-validation)
- **Random Forest Regressor** (with hyperparameter tuning via `RandomizedSearchCV`)

### 🔍 Key Insights:

- **Linear Regression** captured general trends  
  👉 Test R²: **0.57**
- **Lasso Regression** improved generalization  
  👉 Test R²: **0.60**
- **Random Forest Regressor** gave the best performance  
  👉 Test R²: **0.83**

### 📈 Most Influential Features:

- `median_income`
- `ocean_proximity_INLAND`
- `population_per_households`
- `latitude` & `longitude`

Residual analysis confirmed good model fit with minimal bias and acceptable variance.

---

## ⚙️ Tech Stack

- **Programming Language**: Python 🐍

### Libraries:
- `pandas`, `NumPy` – Data loading and processing  
- `matplotlib`, `seaborn` – Visualization  
- `scikit-learn` – ML models, evaluation, and preprocessing  
- `joblib` – Model persistence  
- **Environment**: Google Colab / Jupyter Notebook

---

## 📂 Dataset

- **Source**: [Kaggle - California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Format**: CSV file

---

## 🗂️ Project Structure

```

California-House-Pricing-Prediction/
│
├── data/
│   └── housing.csv                      # Raw dataset
│
├── notebooks/
│   └── HousePricingPrediction.ipynb     # Main analysis and modeling notebook
│
├── plots/                               # Visualizations generated in the notebook
│   ├── linear\_coefficients.png
│   ├── lasso\_coefficients.png
│   ├── rf\_feature\_importance.png
│   └── residuals\_plot.png
│
├── random\_forest\_model.pkl              # Saved trained model
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation

---

## 📊 Evaluation Metrics

| Model               | Test R² | Test MSE |
|--------------------|---------|----------|
| Linear Regression   | 0.5747  | 0.1391   |
| Lasso Regression    | 0.5991  | 0.1311   |
| Random Forest       | 0.8377  | 0.0531   |
| Tuned RF (Best)     | **0.8301** | **0.0556** |

---

## 💡 Key Features

- **Log transformation** and **feature scaling** to improve model performance
- **One-Hot Encoding** for categorical variable (`ocean_proximity`)
- Custom feature engineering:  
  `room_per_households`, `bedroom_per_room`, `population_per_households`
- Residual plots and feature importance visualizations

---

## 🧠 Lessons Learned

- Lasso helped reduce overfitting by penalizing less important features.
- Ensemble methods like Random Forest captured non-linear relationships effectively.
- Preprocessing (especially handling skew and scale) significantly improved performance.

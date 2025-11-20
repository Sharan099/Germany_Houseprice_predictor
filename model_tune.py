# evaluate_and_tune_model.py

# ----------------------------
# Import libraries
# ----------------------------
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(r"H:\Classical_ML\Data\immo_data.csv")

# ----------------------------
# Preprocessing (same as training)
# ----------------------------
top_12 = ['livingSpace', 'noRooms', 'regio1','regio2','regio3','yearConstructed',
          'condition', 'interiorQual', 'balcony', 'hasKitchen', 'floor','totalRent']
df = df[top_12]

df['floor'] = df['floor'].fillna(df['floor'].mode()[0])
df['condition'] = df['condition'].fillna(df['condition'].mode()[0])
df['interiorQual'] = df['interiorQual'].fillna(df['interiorQual'].mode()[0])
df = df.dropna(subset=['totalRent'])

df['yearConstructed'] = df.groupby('regio1')['yearConstructed'] \
                          .transform(lambda x: x.fillna(x.median()))

top_regio3 = df['regio3'].value_counts().nlargest(50).index
df['regio3'] = df['regio3'].apply(lambda x: x if x in top_regio3 else 'other')

# ----------------------------
# Separate target BEFORE encoding
# ----------------------------
y = df['totalRent']
X = df.drop('totalRent', axis=1)

# ----------------------------
# Load saved encoder, scaler, and column order
# ----------------------------
encoder = pickle.load(open(r"H:\Classical_ML\encoder.pkl", "rb"))
scaler = pickle.load(open(r"H:\Classical_ML\scaler.pkl", "rb"))
train_columns = pickle.load(open(r"H:\Classical_ML\train_columns.pkl", "rb"))

# ----------------------------
# New feature engineering
# ----------------------------
X['propertyAge'] = 2025 - X['yearConstructed']
X['livingSpaceRange'] = pd.cut(
    X['livingSpace'],
    bins=[0,50,100,150,200,500],
    labels=[0,1,2,3,4]
)

# ----------------------------
# Encoding
# ----------------------------
X[['regio2','regio3']] = encoder.transform(X[['regio2','regio3']])
X = pd.get_dummies(X, columns=['regio1','condition','interiorQual'], drop_first=True)

# ----------------------------
# Align columns with training
# ----------------------------
for col in train_columns:
    if col not in X.columns:
        X[col] = 0

X = X[train_columns]

# ----------------------------
# Scale numeric cols AFTER column alignment
# ----------------------------
numeric_cols = ['livingSpace','noRooms','yearConstructed','floor']
X[numeric_cols] = scaler.transform(X[numeric_cols])

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Load baseline model
# ----------------------------
baseline_model = joblib.load(r"H:\Classical_ML\best_model.pkl")

# ----------------------------
# Evaluate baseline model
# ----------------------------
y_pred = baseline_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("\n===== BASELINE MODEL METRICS =====")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape*100:.2f}%")

# ----------------------------
# Cross-validation on baseline
# ----------------------------
cv_rmse = -cross_val_score(baseline_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"\nBaseline CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")

# ----------------------------
# Hyperparameter tuning
# ----------------------------
param_grid = {
    'iterations': [500, 1000, 1500],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

cat = CatBoostRegressor(verbose=0, random_state=42)

random_search = RandomizedSearchCV(
    estimator=cat,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
print("\n===== BEST HYPERPARAMETERS FOUND =====")
print(random_search.best_params_)

joblib.dump(best_model, r"H:\Classical_ML\best_model_tuned.pkl")

# ----------------------------
# Evaluate tuned model
# ----------------------------
y_pred_best = best_model.predict(X_test)

mae_b = mean_absolute_error(y_test, y_pred_best)
rmse_b = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_b = r2_score(y_test, y_pred_best)
mape_b = mean_absolute_percentage_error(y_test, y_pred_best)

print("\n===== TUNED MODEL METRICS =====")
print(f"MAE: {mae_b:.2f}")
print(f"RMSE: {rmse_b:.2f}")
print(f"R²: {r2_b:.4f}")
print(f"MAPE: {mape_b*100:.2f}%")

# ----------------------------
# Cross-validation (tuned)
# ----------------------------
cv_rmse_best = -cross_val_score(best_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"\nTuned CV RMSE: {cv_rmse_best.mean():.2f} ± {cv_rmse_best.std():.2f}")

# ----------------------------
# Feature importance plot
# ----------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x=best_model.get_feature_importance(), y=X_train.columns)
plt.title("Feature Importance (Tuned Model)")
plt.show()

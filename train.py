# train_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# ----------------------------
# 1. Load cleaned & scaled data
# ----------------------------
from data_cleaning import scaled_df  # Ensure scaled_df is imported correctly

TARGET = "baseRent"

# ----------------------------
# 2. Define features for prediction
# ----------------------------
features = [
    "geo_plz", "noRooms", "hasKitchen", "floor",
    "garden", "livingSpaceRange", "noRoomsRange",
    "balcony", "lift"
]

# Verify all features exist
missing_cols = [c for c in features if c not in scaled_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in scaled_df: {missing_cols}")

X = scaled_df[features].copy()  # use .copy() to avoid SettingWithCopyWarning
y = scaled_df[TARGET].copy()

# ----------------------------
# 3. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 4. Scale numeric features only
# ----------------------------
numeric_features = ["geo_plz", "noRooms", "floor", "livingSpaceRange", "noRoomsRange"]

scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ----------------------------
# 5. Train Ridge model
# ----------------------------
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ----------------------------
# 6. Feature importance
# ----------------------------
feature_importance = pd.Series(ridge_model.coef_, index=X_train.columns).sort_values(ascending=False)
print("Top Features:")
print(feature_importance)

# ----------------------------
# 7. Save model
# ----------------------------
with open("ridge_model.pkl", "wb") as f:
    pickle.dump(ridge_model, f)

print("✅ Ridge model and scaler saved successfully!")



# X_test, y_test are already defined
y_pred = ridge_model.predict(X_test)

# Evaluate performance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 Score: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


sample = pd.DataFrame({
    "geo_plz": [10115],
    "noRooms": [2],
    "hasKitchen": [1],
    "floor": [1],
    "garden": [0],
    "livingSpaceRange": [1],  # corresponds to your living space bin
    "noRoomsRange": [2],
    "balcony": [0],
    "lift": [0]
})

# Scale numeric features
numeric_cols = ["geo_plz", "noRooms", "floor", "livingSpaceRange", "noRoomsRange"]
sample[numeric_cols] = scaler.transform(sample[numeric_cols])

# Predict
predicted_rent = ridge_model.predict(sample)[0]
print(f"Predicted Base Rent: €{predicted_rent:.2f}")


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Base Rent")
plt.ylabel("Predicted Base Rent")
plt.title("Ridge Model Predictions vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")  # perfect line
plt.show()

# ----------------------------
# Import libraries
# ----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import joblib
import pickle

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(r"H:\Classical_ML\Data\immo_data.csv")

# Keep only top 12 relevant columns
top_12 = ['livingSpace', 'noRooms', 'regio1','regio2','regio3','yearConstructed',
          'condition', 'interiorQual', 'balcony', 'hasKitchen', 'floor','totalRent']
df = df[top_12]

# ----------------------------
# Handle missing values
# ----------------------------
df['floor'] = df['floor'].fillna(df['floor'].mode()[0])
df['condition'] = df['condition'].fillna(df['condition'].mode()[0])
df['interiorQual'] = df['interiorQual'].fillna(df['interiorQual'].mode()[0])
df = df.dropna(subset=['totalRent'])

df['yearConstructed'] = df.groupby('regio1')['yearConstructed'] \
                          .transform(lambda x: x.fillna(x.median()))

# ----------------------------
# Reduce high-cardinality
# ----------------------------
top_regio3 = df['regio3'].value_counts().nlargest(50).index
df['regio3'] = df['regio3'].apply(lambda x: x if x in top_regio3 else 'other')

# ----------------------------
# Encoding
# ----------------------------
# One-hot encode low-cardinality
df = pd.get_dummies(df, columns=['regio1','condition','interiorQual'], drop_first=True)

# Ordinal encode high-cardinality
encoder = OrdinalEncoder()
df[['regio2','regio3']] = encoder.fit_transform(df[['regio2','regio3']])

# Save encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# ----------------------------
# Scale numeric columns
# ----------------------------
num_cols = ['livingSpace', 'noRooms', 'yearConstructed', 'floor']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ----------------------------
# Remove extreme rents
# ----------------------------
df = df[df['totalRent'] < df['totalRent'].quantile(0.99)]

# ----------------------------
# Train-test split
# ----------------------------

X = df.drop('totalRent', axis=1)
y = df['totalRent']

# Save final training column order
with open(r"H:\Classical_ML\train_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train CatBoost model
# ----------------------------
cat_model = CatBoostRegressor(
    verbose=0,
    iterations=1500,
    learning_rate=0.05,
    depth=8,
    loss_function='RMSE'
)
cat_model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = cat_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"CatBoost Results:\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# ----------------------------
# Save final best model
# ----------------------------
joblib.dump(cat_model, "best_model.pkl")
print("Model saved as best_model.pkl")

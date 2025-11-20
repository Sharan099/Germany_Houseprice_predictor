# app.py
import streamlit as st
import pandas as pd
import pickle
import joblib

# ----------------------------
# Load saved objects
# ----------------------------
best_model = joblib.load(r"H:\Classical_ML\best_model.pkl")

with open(r"H:\Classical_ML\encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open(r"H:\Classical_ML\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(r"H:\Classical_ML\train_columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

# ----------------------------
# Streamlit page UI
# ----------------------------
st.set_page_config(page_title="🏠 Germany Rent Prediction", layout="centered")
st.title("🏠 Predict Base Rent in Germany")
st.markdown("Fill the details below to estimate the rent.")

# ----------------------------
# User Input Function
# ----------------------------
def user_input_features():

    # EXACT categories seen in training (important)
    regio1 = st.selectbox("State / regio1", [
        "Bayern", "Berlin", "Hessen", "Sachsen", "Nordrhein_Westfalen",
        "Rheinland_Pfalz", "Bremen", "other"
    ])

    condition = st.selectbox("Condition", [
        "well_kept", "refurbished", "fully_renovated", "mint_condition", "first_time_use"
    ])

    interiorQual = st.selectbox("Interior Quality", [
        "normal", "sophisticated", "simple", "luxury"
    ])

    # Encoder categories from training
    regio2 = st.selectbox("District / regio2", encoder.categories_[0])
    regio3 = st.selectbox("Neighborhood / regio3", encoder.categories_[1])

    livingSpace = st.number_input("Living Space (sqm)", min_value=10, max_value=500, value=50)
    noRooms = st.number_input("Number of Rooms", min_value=1, max_value=12, value=2)
    floor = st.number_input("Floor", min_value=0, max_value=50, value=1)
    yearConstructed = st.number_input("Year Constructed", min_value=1800, max_value=2025, value=2000)
    balcony = st.selectbox("Balcony", [0, 1])
    hasKitchen = st.selectbox("Has Kitchen", [0, 1])

    data = {
        "livingSpace": livingSpace,
        "noRooms": noRooms,
        "floor": floor,
        "yearConstructed": yearConstructed,
        "balcony": balcony,
        "hasKitchen": hasKitchen,
        "regio1": regio1,
        "regio2": regio2,
        "regio3": regio3,
        "condition": condition,
        "interiorQual": interiorQual
    }

    return pd.DataFrame(data, index=[0])


# ----------------------------
# Get input
# ----------------------------
input_df = user_input_features()

# ----------------------------
# Ordinal Encode regio2, regio3
# ----------------------------
input_df[["regio2", "regio3"]] = encoder.transform(
    input_df[["regio2", "regio3"]]
)

# ----------------------------
# One-hot encode low-cardinality
# ----------------------------
input_df = pd.get_dummies(
    input_df,
    columns=["regio1", "condition", "interiorQual"],
    drop_first=True
)

# ----------------------------
# Add missing columns
# ----------------------------
for col in train_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[train_columns]

# ----------------------------
# Scale numeric columns
# ----------------------------
num_cols = ['livingSpace', 'noRooms', 'yearConstructed', 'floor']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# ----------------------------
# Predict
# ----------------------------
prediction = best_model.predict(input_df)[0]

st.subheader("Predicted Base Rent (€)")
st.success(f"€ {prediction:.2f}")

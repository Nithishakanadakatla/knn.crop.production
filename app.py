import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# APP SETUP
# -----------------------------
st.set_page_config(layout="wide")
st.title("🌾 Crop Production Prediction (Fixed UI Version)")

# -----------------------------
# UPLOAD DATA
# -----------------------------
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # CLEAN MISSING VALUES
    # -----------------------------
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mean())

    st.success("Missing values handled")

    # -----------------------------
    # TARGET COLUMN
    # -----------------------------
    target = "Production"

    if target not in df.columns:
        st.error("Production column not found!")
        st.stop()

    # -----------------------------
    # ENCODING
    # -----------------------------
    df_encoded = df.copy()
    encoders = {}

    categorical_cols = []

    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
            categorical_cols.append(col)

    # -----------------------------
    # FEATURES & TARGET
    # -----------------------------
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # -----------------------------
    # TRAIN TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    y_train = np.nan_to_num(y_train)
    y_test = np.nan_to_num(y_test)

    # -----------------------------
    # MODEL
    # -----------------------------
    k = st.slider("Select K Value", 1, 20, 5)

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # -----------------------------
    # METRICS
    # -----------------------------
    st.subheader("📈 Model Performance")

    st.write("MSE:", round(mean_squared_error(y_test, y_pred), 2))
    st.write("R² Score:", round(r2_score(y_test, y_pred), 2))

    # -----------------------------
    # 🔮 PREDICTION UI (FIXED)
    # -----------------------------
    st.subheader("🔮 Predict Production")

    input_data = []

    for col in X.columns:

        # 🔴 CATEGORICAL → DROPDOWN
        if col in encoders:

            options = list(encoders[col].classes_)
            selected = st.selectbox(col, options)

            encoded = encoders[col].transform([selected])[0]
            input_data.append(encoded)

        # 🔵 NUMERIC → NUMBER INPUT
        else:
            val = st.number_input(col, value=float(df[col].mean()))
            input_data.append(val)

    # -----------------------------
    # PREDICT BUTTON
    # -----------------------------
    if st.button("Predict"):

        input_array = np.array(input_data).reshape(1, -1)
        input_array = scaler.transform(input_array)
        input_array = np.nan_to_num(input_array)

        result = model.predict(input_array)

        st.success(f"🌾 Predicted Production: {result[0]:.2f}")

else:
    st.info("👆 Upload your CSV file to start")
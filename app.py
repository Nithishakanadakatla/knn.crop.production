import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("🌾 KNN Crop Production Predictor (Smart UI)")

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
    # MISSING VALUES
    # -----------------------------
    st.subheader("🧹 Missing Values Handling")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mean())

    st.success("Missing values handled")

    # -----------------------------
    # TARGET
    # -----------------------------
    target = st.selectbox("Select Target Column", df.columns)

    # -----------------------------
    # ENCODING
    # -----------------------------
    df_encoded = df.copy()
    encoders = {}

    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le

    # -----------------------------
    # OUTLIERS (SAFE)
    # -----------------------------
    num_cols = df_encoded.select_dtypes(include=np.number).columns

    for col in num_cols:
        df_encoded[col] = df_encoded[col].clip(
            df_encoded[col].quantile(0.05),
            df_encoded[col].quantile(0.95)
        )

    # -----------------------------
    # FEATURES
    # -----------------------------
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # -----------------------------
    # SPLIT
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
    # PREDICTION SECTION (IMPROVED UI)
    # -----------------------------
    st.subheader("🔮 Predict Production (Smart Input UI)")

    user_input = []

    for col in X.columns:

        # 🟢 If categorical → dropdown
        if col in encoders:

            options = list(encoders[col].classes_)
            selected = st.selectbox(f"Select {col}", options)

            encoded_value = encoders[col].transform([selected])[0]
            user_input.append(encoded_value)

        # 🔵 numeric → number input
        else:
            val = st.number_input(f"Enter {col}", value=float(df[col].mean()))
            user_input.append(val)

    # -----------------------------
    # Predict Button
    # -----------------------------
    if st.button("Predict Production"):

        input_array = np.array(user_input).reshape(1, -1)
        input_array = scaler.transform(input_array)
        input_array = np.nan_to_num(input_array)

        prediction = model.predict(input_array)

        st.success(f"🌾 Predicted Production: {prediction[0]:.2f}")

else:
    st.info("👆 Upload dataset to start")
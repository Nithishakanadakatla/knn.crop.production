import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(layout="wide")
st.title("🌾 KNN Regression - Crop Production ")

# -----------------------------
# Upload Dataset
# -----------------------------
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    # Clean column names
    df.columns = df.columns.str.strip()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📌 Dataset Summary")
    st.write(df.describe(include="all"))

    # -----------------------------
    # Missing Values Handling (SAFE)
    # -----------------------------
    st.subheader("🧹 Missing Values Handling")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mean())

    st.success("Missing values handled safely")

    # -----------------------------
    # Select Target
    # -----------------------------
    target = st.selectbox("Select Target Column", df.columns)

    # -----------------------------
    # Encoding
    # -----------------------------
    st.subheader("🔤 Encoding Categorical Data")

    df_encoded = df.copy()
    encoders = {}

    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le

    st.success("Encoding completed")

    st.dataframe(df_encoded.head())

    # -----------------------------
    # Outlier Handling (SAFE - NO ROW DELETION)
    # -----------------------------
    st.subheader("📊 Outlier Handling (Safe Method)")

    num_cols = df_encoded.select_dtypes(include=np.number).columns

    for col in num_cols:
        lower = df_encoded[col].quantile(0.05)
        upper = df_encoded[col].quantile(0.95)
        df_encoded[col] = df_encoded[col].clip(lower, upper)

    st.success("Outliers handled safely (no rows removed)")

    # -----------------------------
    # Features & Target
    # -----------------------------
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    if len(X) == 0:
        st.error("Dataset became empty after preprocessing. Please check data.")
        st.stop()

    # -----------------------------
    # Train Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Model Selection
    # -----------------------------
    k = st.slider("Select K Value", 1, 20, 5)

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # -----------------------------
    # Metrics
    # -----------------------------
    st.subheader("📈 Model Performance")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", round(mse, 2))
    col2.metric("RMSE", round(rmse, 2))
    col3.metric("R² Score", round(r2, 2))



    # -----------------------------
    # Prediction Section
    # -----------------------------
    st.subheader("🔮 Predict Production")

    user_input = []

    for col in X.columns:
        if col in encoders:
            val = st.selectbox(col, encoders[col].classes_)
            val = encoders[col].transform([val])[0]
        else:
            val = st.number_input(col, value=float(df_encoded[col].mean()))
        user_input.append(val)

    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)

    if st.button("Predict"):
        result = model.predict(user_input)
        st.success(f"🌾 Predicted Value: {result[0]:.2f}")

else:
    st.info("👆 Upload CSV file to start")
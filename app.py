import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("🌾 KNN Regression - Crop production")

# -------------------------------
# 📂 Upload Dataset
# -------------------------------
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("📊 Original Dataset")
    st.dataframe(df.head())

    # -------------------------------
    # 🧹 CLEANING
    # -------------------------------

    # Remove commas (important)
    df = df.replace(',', '', regex=True)

    # Drop missing values
    df = df.dropna()

    # -------------------------------
    # 🔤 ENCODING
    # -------------------------------
    le_dict = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    # -------------------------------
    # 🔢 FORCE NUMERIC (MAIN FIX)
    # -------------------------------
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    st.subheader("🔍 Data Types After Processing")
    st.write(df.dtypes)

    # -------------------------------
    # 🎯 SELECT TARGET
    # -------------------------------
    target = st.selectbox("Select Target Column", df.columns)
    # -------------------------------
    # 📊 Handle Outliers (CAPPING METHOD)
    # -------------------------------
    st.subheader("📊 Handling Outliers (Safe Method)")
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
        st.write("✅ Outliers handled without removing rows")

    # -------------------------------
    # 🎯 FEATURES & TARGET
    # -------------------------------
    X = df.drop(target, axis=1)
    y = df[target]

    # -------------------------------
    # 🚨 SAFETY CHECK (NO ERROR)
    # -------------------------------
    if X.select_dtypes(include=['object']).shape[1] > 0:
        st.error("❌ Still contains non-numeric columns!")
        st.stop()

    # -------------------------------
    # ✂️ TRAIN TEST SPLIT
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # ⚖️ SCALING
    # -------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # 🔢 SELECT K
    # -------------------------------
    k = st.slider("Select K Value", 1, 20, 5)

    # -------------------------------
    # 🤖 TRAIN MODEL
    # -------------------------------
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # -------------------------------
    # 📈 METRICS
    # -------------------------------
    st.subheader("📈 Model Performance")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")

    # -------------------------------
    # 🔮 PREDICTION UI
    # -------------------------------
    st.subheader("🔮 Make Prediction")

    user_input = []

    for col in X.columns:
        if col in le_dict:
            val = st.selectbox(col, le_dict[col].classes_)
            val = le_dict[col].transform([val])[0]
        else:
            val = st.number_input(col, value=float(X[col].mean()))

        user_input.append(val)

    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)

    if st.button("Predict"):
        result = model.predict(user_input)
        st.success(f"🌱 Predicted Value: {result[0]:.2f}")

else:
    st.warning("⚠️ Please upload a dataset to continue")
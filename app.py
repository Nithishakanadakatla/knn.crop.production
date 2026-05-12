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

# -----------------------------
# 📂 Upload Dataset
# -----------------------------
file = st.file_uploader("Upload your CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)

    # -----------------------------
    # 📊 Show Dataset
    # -----------------------------
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    st.subheader("📌 Dataset Info")
    st.write(df.describe())

    # -----------------------------
    # 🧹 Missing Values
    # -----------------------------
    st.subheader("🧹 Missing Values")
    st.write(df.isnull().sum())

    df = df.dropna()

    # -----------------------------
    # 🎯 Select Target
    # -----------------------------
    target = st.selectbox("Select Target Column", df.columns)

    # -----------------------------
    # 🔤 Encoding
    # -----------------------------
    st.subheader("🔤 Encoding Categorical Columns")

    df_encoded = df.copy()
    encoders = {}

    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            encoders[col] = le

    st.write("Encoded Data Preview")
    st.dataframe(df_encoded.head())

    # -----------------------------
    # 📊 Outlier Visualization
    # -----------------------------
    st.subheader("📊 Outlier Detection (Boxplot)")

    num_cols = df_encoded.select_dtypes(include=np.number).columns

    fig, ax = plt.subplots()
    sns.boxplot(data=df_encoded[num_cols], ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # 🧹 Remove Outliers (IQR)
    # -----------------------------
    Q1 = df_encoded[num_cols].quantile(0.25)
    Q3 = df_encoded[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    df_clean = df_encoded[~((df_encoded[num_cols] < (Q1 - 1.5 * IQR)) |
                             (df_encoded[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    st.write("Shape after removing outliers:", df_clean.shape)

    # -----------------------------
    # 🎯 Features & Target
    # -----------------------------
    X = df_clean.drop(target, axis=1)
    y = df_clean[target]

    # -----------------------------
    # ✂️ Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # ⚖️ Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # 🔢 Select K
    # -----------------------------
    k = st.slider("Select K value", 1, 20, 5)

    # -----------------------------
    # 🤖 Train Model
    # -----------------------------
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # -----------------------------
    # 📈 Metrics
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
    # 🔮 Prediction Section
    # -----------------------------
    st.subheader("🔮 Make Prediction")

    user_input = []

    for col in X.columns:
        if col in encoders:
            val = st.selectbox(col, encoders[col].classes_)
            val = encoders[col].transform([val])[0]
        else:
            val = st.number_input(col, value=float(df[col].mean()))
        user_input.append(val)

    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)

    if st.button("Predict"):
        result = model.predict(user_input)
        st.success(f"Predicted Value: {result[0]:.2f}")

else:
    st.info("👆 Upload your dataset to start")
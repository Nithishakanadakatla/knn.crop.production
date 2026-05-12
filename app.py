import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("🌾 KNN Crop Production Prediction App")

# -----------------------------
# UPLOAD DATASET
# -----------------------------
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    st.subheader("🧹 Missing Value Handling")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mean())

    st.success("Missing values cleaned safely")

    # -----------------------------
    # SELECT TARGET
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

    st.success("Encoding completed")

    # -----------------------------
    # SAFE OUTLIER HANDLING
    # -----------------------------
    num_cols = df_encoded.select_dtypes(include=np.number).columns

    for col in num_cols:
        lower = df_encoded[col].quantile(0.05)
        upper = df_encoded[col].quantile(0.95)
        df_encoded[col] = df_encoded[col].clip(lower, upper)

    # -----------------------------
    # FEATURES & TARGET
    # -----------------------------
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    if len(X) == 0:
        st.error("Dataset became empty after preprocessing.")
        st.stop()

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

    # -----------------------------
    # 🔥 FINAL SAFETY CLEANING (IMPORTANT)
    # -----------------------------
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    y_train = np.nan_to_num(y_train)
    y_test = np.nan_to_num(y_test)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # -----------------------------
    # MODEL SELECTION
    # -----------------------------
    k = st.slider("Select K Value", 1, 20, 5)

    if k >= len(X_train):
        st.error("K value is too large for dataset size!")
        st.stop()

    model = KNeighborsRegressor(n_neighbors=k)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -----------------------------
    # METRICS
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
    # PREDICTION SECTION (SAFE UI)
    # -----------------------------
    st.subheader("🔮 Predict Production")

    user_input = []

    for col in X.columns:

        if col in encoders:
            options = list(encoders[col].classes_)
            val = st.selectbox(f"{col}", options)
            val = encoders[col].transform([val])[0]
        else:
            val = st.number_input(f"{col}", value=float(df_encoded[col].mean()))

        user_input.append(val)

    # -----------------------------
    # PREDICT BUTTON
    # -----------------------------
    if st.button("Predict"):

        input_array = np.array(user_input).reshape(1, -1)
        input_array = scaler.transform(input_array)

        input_array = np.nan_to_num(input_array)

        result = model.predict(input_array)

        st.success(f"🌾 Predicted Production: {result[0]:.2f}")

else:
    st.info("👆 Upload CSV file to start")
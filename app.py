import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------
# Streamlit Setup
# ---------------------------
st.set_page_config(page_title="Crop Production Predictor", layout="wide")
st.title("🌾 Crop Production Prediction App")

# ---------------------------
# Upload Dataset
# ---------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Clean column names
    # ---------------------------
    df.columns = df.columns.str.strip()

    # ---------------------------
    # Handle missing values safely
    # ---------------------------
    st.write("🔧 Handling missing values...")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mean())

    st.success("Missing values handled")

    # ---------------------------
    # Encode categorical columns
    # ---------------------------
    st.write("🔤 Encoding categorical features...")

    encoders = {}

    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    st.success("Encoding completed")

    # ---------------------------
    # Select target column
    # ---------------------------
    if "Production" in df.columns:
        target = "Production"
    else:
        target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target]

    st.write("📌 Features shape:", X.shape)

    # ---------------------------
    # Train-test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # Scaling
    # ---------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------
    # Model training
    # ---------------------------
    st.write("🤖 Training model...")

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ---------------------------
    # Evaluation
    # ---------------------------
    y_pred = model.predict(X_test)

    st.subheader("📈 Model Performance")
    st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    st.write("R² Score:", round(r2_score(y_test, y_pred), 2))

    # ---------------------------
    # Prediction section
    # ---------------------------
    st.subheader("🔮 Predict Production")

    input_data = []

    for col in X.columns:
        val = st.number_input(f"Enter {col}", value=float(X[col].mean()))
        input_data.append(val)

    if st.button("Predict"):

        input_array = np.array(input_data).reshape(1, -1)
        input_array = scaler.transform(input_array)

        prediction = model.predict(input_array)

        st.success(f"🌾 Predicted Production: {prediction[0]:.2f}")

else:
    st.info("⬆ Upload your dataset to start")
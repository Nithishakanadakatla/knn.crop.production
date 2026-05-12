import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="KNN Crop Predictor", layout="wide")
st.title("🌾 KNN Crop Prediction App (Clean Version)")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(data.head())

    # ---------------------------
    # Handle Missing Values
    # ---------------------------
    st.write("🔧 Handling missing values...")

    for col in data.columns:
        if data[col].dtype == "object":
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].mean(), inplace=True)

    st.success("Missing values handled")

    # ---------------------------
    # Encode Categorical Data
    # ---------------------------
    st.write("🔤 Encoding categorical features...")

    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    st.success("Encoding completed")

    # ---------------------------
    # Split Features & Target
    # ---------------------------
    target_column = data.columns[-1]
    X = data.drop(columns=[target_column])
    y = data[target_column]

    st.write("📌 Features shape:", X.shape)

    # ---------------------------
    # Safe Outlier Handling (Clipping instead of deleting rows)
    # ---------------------------
    st.write("📊 Handling Outliers (Safe Method)")

    for col in X.columns:
        if X[col].dtype != "object":
            lower = X[col].quantile(0.05)
            upper = X[col].quantile(0.95)
            X[col] = X[col].clip(lower, upper)

    st.success("Outliers handled safely (no rows removed)")

    # ---------------------------
    # Safety Check
    # ---------------------------
    if len(X) == 0:
        st.error("Dataset is empty after preprocessing. Please check your data.")
        st.stop()

    # ---------------------------
    # Train-Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # Feature Scaling
    # ---------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------
    # Model Training
    # ---------------------------
    st.write("🤖 Training KNN Model...")

    k = st.slider("Select K value", 1, 15, 5)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # ---------------------------
    # Prediction
    # ---------------------------
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.subheader("📈 Model Results")
    st.write("Accuracy:", round(acc * 100, 2), "%")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # ---------------------------
    # Single Prediction UI
    # ---------------------------
    st.subheader("🔮 Make a Prediction")

    input_data = []
    for col in X.columns:
        val = st.number_input(f"Enter {col}", value=float(X[col].mean()))
        input_data.append(val)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        input_array = scaler.transform(input_array)

        prediction = model.predict(input_array)

        st.success(f"Predicted Output: {prediction[0]}")

else:
    st.info("⬆ Upload a CSV file to start")

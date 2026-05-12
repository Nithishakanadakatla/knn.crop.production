# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="KNN Regression - Crop Production",
    layout="wide"
)

st.title("🌾 KNN Regression - Crop Production")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # ---------------- LOAD DATA ----------------

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------------- MISSING VALUES ----------------

    st.subheader("🧹 Missing Values")

    missing_values = df.isnull().sum()

    st.dataframe(
        missing_values.reset_index().rename(
            columns={"index": "Column", 0: "Missing Values"}
        ),
        use_container_width=True
    )

    # Remove missing rows
    df = df.dropna()

    # ---------------- ENCODING ----------------

    st.subheader("🔤 Encoding Categorical Columns")

    encoded_df = df.copy()

    label_encoders = {}

    for col in encoded_df.columns:

        # Encode object columns
        if encoded_df[col].dtype == "object":

            le = LabelEncoder()

            encoded_df[col] = le.fit_transform(
                encoded_df[col].astype(str)
            )

            label_encoders[col] = le

    # Convert all columns to numeric
    for col in encoded_df.columns:

        encoded_df[col] = pd.to_numeric(
            encoded_df[col],
            errors="coerce"
        )

    # Remove NaN after conversion
    encoded_df = encoded_df.dropna()

    st.write("Encoded Dataset Preview")

    st.dataframe(
        encoded_df.head(),
        use_container_width=True
    )

    # ---------------- TARGET COLUMN ----------------

    st.subheader("🎯 Select Target Column")

    numeric_columns = encoded_df.columns.tolist()

    target_column = st.selectbox(
        "Select Target Column",
        numeric_columns
    )

    # ---------------- OUTLIER DETECTION ----------------

    st.subheader("📈 Outlier Detection (Boxplot)")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.boxplot(encoded_df[target_column])

    ax.set_title(f"Boxplot of {target_column}")

    st.pyplot(fig)

    # ---------------- REMOVE OUTLIERS ----------------

    Q1 = encoded_df[target_column].quantile(0.25)

    Q3 = encoded_df[target_column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR

    clean_df = encoded_df[
        (encoded_df[target_column] >= lower_bound) &
        (encoded_df[target_column] <= upper_bound)
    ]

    st.success(f"Shape after removing outliers: {clean_df.shape}")

    # ---------------- FEATURES & TARGET ----------------

    X = clean_df.drop(columns=[target_column])

    y = clean_df[target_column]

    # FINAL FIX FOR ERROR
    X = X.astype(float)

    y = y.astype(float)

    # ---------------- TRAIN TEST SPLIT ----------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------- K VALUE ----------------

    st.subheader("🎚 Select K Value")

    k_value = st.slider(
        "K Value",
        min_value=1,
        max_value=15,
        value=5
    )

    # ---------------- MODEL ----------------

    model = KNeighborsRegressor(
        n_neighbors=k_value
    )

    model.fit(X_train, y_train)

    # ---------------- PREDICTION ----------------

    y_pred = model.predict(X_test)

    # ---------------- METRICS ----------------

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    # ---------------- MODEL PERFORMANCE ----------------

    st.subheader("📉 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MSE", round(mse, 2))

    with col2:
        st.metric("RMSE", round(rmse, 2))

    with col3:
        st.metric("R² Score", round(r2, 2))

    # ---------------- MANUAL PREDICTION ----------------

    st.subheader("🧑‍🌾 Make Prediction")

    input_data = {}

    for column in X.columns:

        value = st.number_input(
            f"Enter {column}",
            value=float(X[column].mean())
        )

        input_data[column] = value

    # ---------------- PREDICT BUTTON ----------------

    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)

        st.success(
            f"Predicted Value: {prediction[0]:.2f}"
        )

else:
    st.info("Please upload a CSV file.")
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

    missing_df = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"])

    st.dataframe(missing_df, use_container_width=True)

    # Remove missing rows
    df = df.dropna()

    # ---------------- TARGET COLUMN ----------------
    st.subheader("🎯 Select Target Column")

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_columns) == 0:
        st.error("No numeric columns found in dataset.")
        st.stop()

    target_column = st.selectbox(
        "Select Target Column",
        numeric_columns
    )

    # ---------------- ENCODING ----------------
    st.subheader("🔤 Encoding Categorical Columns")

    encoded_df = df.copy()

    label_encoders = {}

    for col in encoded_df.columns:

        if encoded_df[col].dtype == "object":

            le = LabelEncoder()

            encoded_df[col] = le.fit_transform(
                encoded_df[col].astype(str)
            )

            label_encoders[col] = le

    st.write("Encoded Data Preview")

    st.dataframe(
        encoded_df.head(),
        use_container_width=True
    )

    # ---------------- CONVERT TARGET TO NUMERIC ----------------
    encoded_df[target_column] = pd.to_numeric(
        encoded_df[target_column],
        errors="coerce"
    )

    encoded_df = encoded_df.dropna()

    # ---------------- OUTLIER DETECTION ----------------
    st.subheader("📈 Outlier Detection (Boxplot)")

    fig, ax = plt.subplots(figsize=(8, 4))

    # FIXED ERROR HERE
    numeric_target = encoded_df[target_column].astype(float)

    ax.boxplot(numeric_target)

    ax.set_title(f"Boxplot of {target_column}")

    st.pyplot(fig)

    # ---------------- REMOVE OUTLIERS ----------------
    Q1 = numeric_target.quantile(0.25)
    Q3 = numeric_target.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    clean_df = encoded_df[
        (numeric_target >= lower_bound) &
        (numeric_target <= upper_bound)
    ]

    st.success(f"Shape after removing outliers: {clean_df.shape}")

    # ---------------- FEATURES & TARGET ----------------
    X = clean_df.drop(columns=[target_column])

    y = clean_df[target_column]

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

    # ---------------- PREDICTION SECTION ----------------
    st.subheader("🧑‍🌾 Make Prediction")

    input_data = {}

    for column in X.columns:

        # categorical columns
        if column in label_encoders:

            options = list(
                label_encoders[column].classes_
            )

            selected_option = st.selectbox(
                column,
                options
            )

            encoded_value = label_encoders[column].transform(
                [selected_option]
            )[0]

            input_data[column] = encoded_value

        # numeric columns
        else:

            value = st.number_input(
                column,
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
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- PAGE SETTINGS ----------------

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

    try:

        # ---------------- READ CSV ----------------

        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # ---------------- REMOVE NULL VALUES ----------------

        df = df.dropna()

        # ---------------- SHOW MISSING VALUES ----------------

        st.subheader("🧹 Missing Values")

        missing_df = pd.DataFrame(
            df.isnull().sum(),
            columns=["Missing Values"]
        )

        st.dataframe(missing_df)

        # ---------------- ENCODE CATEGORICAL COLUMNS ----------------

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

        # ---------------- FORCE ALL DATA TO NUMERIC ----------------

        encoded_df = encoded_df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Remove any remaining NaN rows
        encoded_df = encoded_df.dropna()

        st.write("Encoded Data Preview")

        st.dataframe(
            encoded_df.head(),
            use_container_width=True
        )

        # ---------------- TARGET COLUMN ----------------

        st.subheader("🎯 Select Target Column")

        target_column = st.selectbox(
            "Select Target Column",
            encoded_df.columns
        )

        # ---------------- BOXPLOT ----------------

        st.subheader("📈 Outlier Detection (Boxplot)")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.boxplot(
            encoded_df[target_column].values.astype(float)
        )

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

        st.success(
            f"Shape after removing outliers: {clean_df.shape}"
        )

        # ---------------- FEATURES & TARGET ----------------

        X = clean_df.drop(columns=[target_column])

        y = clean_df[target_column]

        # ---------------- FINAL FIX ----------------
        # THIS FIXES YOUR K VALUE ERROR

        X = X.astype(np.float64)

        y = y.astype(np.float64)

        # Remove infinity values
        X = X.replace([np.inf, -np.inf], np.nan)

        y = y.replace([np.inf, -np.inf], np.nan)

        # Remove NaN again
        valid_rows = ~(X.isna().any(axis=1) | y.isna())

        X = X[valid_rows]

        y = y[valid_rows]

        # ---------------- TRAIN TEST SPLIT ----------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ---------------- K VALUE ----------------

        st.subheader("🎚 Select K Value")

        max_k = min(15, len(X_train))

        k_value = st.slider(
            "K Value",
            min_value=1,
            max_value=max_k,
            value=min(5, max_k)
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

        # ---------------- PERFORMANCE ----------------

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

            input_df = input_df.astype(np.float64)

            prediction = model.predict(input_df)

            st.success(
                f"Predicted Value: {prediction[0]:.2f}"
            )

    except Exception as e:

        st.error(f"Error: {e}")

else:

    st.info("Please upload a CSV file.")
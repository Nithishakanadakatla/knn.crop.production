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

    try:

        # ---------------- READ CSV ----------------

        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # ---------------- REMOVE EMPTY ROWS ----------------

        df = df.dropna()

        # ---------------- SHOW MISSING VALUES ----------------

        st.subheader("🧹 Missing Values")

        st.dataframe(
            pd.DataFrame(
                df.isnull().sum(),
                columns=["Missing Values"]
            ),
            use_container_width=True
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

        # ---------------- KEEP ONLY NUMERIC COLUMNS ----------------

        encoded_df = encoded_df.select_dtypes(
            include=[np.number]
        )

        st.write("Encoded Dataset Preview")

        st.dataframe(
            encoded_df.head(),
            use_container_width=True
        )

        # ---------------- CHECK DATASET ----------------

        if encoded_df.shape[0] == 0:

            st.error("Dataset is empty.")

            st.stop()

        # ---------------- TARGET COLUMN ----------------

        st.subheader("🎯 Select Target Column")

        target_column = st.selectbox(
            "Select Target Column",
            encoded_df.columns
        )

        # ---------------- BOXPLOT ----------------

        st.subheader("📈 Outlier Detection")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.boxplot(encoded_df[target_column])

        ax.set_title(
            f"Boxplot of {target_column}"
        )

        st.pyplot(fig)

        # ---------------- NO OUTLIER REMOVAL ----------------
        # THIS IS THE FIX

        clean_df = encoded_df.copy()

        st.success(
            f"Dataset Shape: {clean_df.shape}"
        )

        # ---------------- FEATURES & TARGET ----------------

        X = clean_df.drop(
            columns=[target_column]
        )

        y = clean_df[target_column]

        # ---------------- CONVERT TO FLOAT ----------------

        X = X.astype(float)

        y = y.astype(float)

        # ---------------- CHECK EMPTY ----------------

        if len(X) < 2:

            st.error(
                "Not enough data for training."
            )

            st.stop()

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

        if max_k < 1:
            max_k = 1

        k_value = st.slider(
            "K Value",
            min_value=1,
            max_value=max_k,
            value=1
        )

        # ---------------- MODEL ----------------

        model = KNeighborsRegressor(
            n_neighbors=k_value
        )

        model.fit(X_train, y_train)

        # ---------------- PREDICTIONS ----------------

        y_pred = model.predict(X_test)

        # ---------------- METRICS ----------------

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            y_pred
        )

        # ---------------- RESULTS ----------------

        st.subheader("📉 Model Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "MSE",
                round(mse, 2)
            )

        with col2:
            st.metric(
                "RMSE",
                round(rmse, 2)
            )

        with col3:
            st.metric(
                "R² Score",
                round(r2, 2)
            )

        # ---------------- MANUAL PREDICTION ----------------

        st.subheader("🧑‍🌾 Make Prediction")

        input_data = {}

        for column in X.columns:

            input_data[column] = st.number_input(
                f"Enter {column}",
                value=float(X[column].mean())
            )

        # ---------------- PREDICT BUTTON ----------------

        if st.button("Predict"):

            input_df = pd.DataFrame(
                [input_data]
            )

            prediction = model.predict(
                input_df
            )

            st.success(
                f"Predicted Value: {prediction[0]:.2f}"
            )

    except Exception as e:

        st.error(f"Error: {e}")

else:

    st.info("Please upload a CSV file.")
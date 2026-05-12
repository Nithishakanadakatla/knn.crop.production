import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("🌾 Crop Production Prediction - KNN App")

# -----------------------------
# UPLOAD DATA
# -----------------------------
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # MISSING VALUES HANDLING
    # -----------------------------
    st.subheader("🧹 Handling Missing Values")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mean())

    st.success("Missing values handled successfully")

    # -----------------------------
    # TARGET COLUMN
    # -----------------------------
    target = "Production"

    if target not in df.columns:
        st.error("❌ Production column not found in dataset")
        st.stop()

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
    # FEATURES & TARGET
    # -----------------------------
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

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

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    y_train = np.nan_to_num(y_train)
    y_test = np.nan_to_num(y_test)

    # -----------------------------
    # MODEL
    # -----------------------------
    k = st.slider("Select K Value", 1, 20, 5)

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # -----------------------------
    # METRICS
    # -----------------------------
    st.subheader("📈 Model Performance")

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("MSE", round(mse, 2))
    col2.metric("R² Score", round(r2, 2))

    # -----------------------------
    # 📉 K vs ERROR GRAPH
    # -----------------------------
    st.subheader("📉 Error vs K")

    errors = []
    for i in range(1, 21):
        temp_model = KNeighborsRegressor(n_neighbors=i)
        temp_model.fit(X_train, y_train)
        pred = temp_model.predict(X_test)
        errors.append(mean_squared_error(y_test, pred))

    st.line_chart(pd.DataFrame({"K": range(1, 21), "MSE": errors}).set_index("K"))

    # -----------------------------
    # 🔮 PREDICTION UI (FORM STYLE LIKE YOUR IMAGE)
    # -----------------------------
    st.subheader("🔮 Make Prediction")

    with st.form("prediction_form"):

        district = st.selectbox("District_Name", encoders["District_Name"].classes_)
        season = st.selectbox("Season", encoders["Season"].classes_)
        crop = st.selectbox("Crop", encoders["Crop"].classes_)

        crop_year = st.number_input("Crop_Year", value=2005.0, step=1.0)
        area = st.number_input("Area", value=float(df["Area"].mean()), step=10.0)

        submit = st.form_submit_button("Predict")

    # -----------------------------
    # PREDICTION LOGIC
    # -----------------------------
    if submit:

        district = encoders["District_Name"].transform([district])[0]
        season = encoders["Season"].transform([season])[0]
        crop = encoders["Crop"].transform([crop])[0]

        input_data = np.array([[district, crop_year, season, crop, area]])

        input_data = scaler.transform(input_data)
        input_data = np.nan_to_num(input_data)

        result = model.predict(input_data)

        st.success(f"🌾 Predicted Production: {result[0]:.2f}")

else:
    st.info("👆 Upload CSV file to start")
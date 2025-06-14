import streamlit as st
import pandas as pd
import joblib

# Load models and encoders
rf_model = joblib.load("iris_random_forest.pkl")
svm_model = joblib.load("iris_svm.pkl")
label_encoder = joblib.load("label_encoder.pkl")
accuracies = joblib.load("model_accuracies.pkl")

# Set page configuration
st.set_page_config(page_title="Iris Classifier", layout="centered", page_icon="🌸")

# Custom CSS styling
st.markdown("""
    <style>
    html, body, [data-testid="stApp"] {
        background: linear-gradient(to right, #f0f8ff, #e6f0fa);
    }
    .main {
        padding: 2rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# App Title and Description
st.title("🌸 Iris Species Classifier")
st.write("Predict the species of an Iris flower using **Random Forest** and **SVM** models.")

# Input form
st.subheader("📏 Enter Flower Measurements")
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.5, 0.1)

    with col2:
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

    submitted = st.form_submit_button("🔍 Predict")

# Prediction and output
if submitted:
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    )

    rf_pred = rf_model.predict(input_data)
    svm_pred = svm_model.predict(input_data)

    rf_result = label_encoder.inverse_transform(rf_pred)[0]
    svm_result = label_encoder.inverse_transform(svm_pred)[0]

    st.subheader("🔎 Prediction Results")
    st.info(f"🌲 Random Forest: {rf_result}")
    st.info(f"🔀 SVM: {svm_result}")

    # Display model accuracies on main page
    st.subheader("📊 Model Accuracy")
    st.success(f"🌲 Random Forest Accuracy: **{accuracies['rf'] * 100:.2f}%**")
    st.success(f"🔀 SVM Accuracy: **{accuracies['svm'] * 100:.2f}%**")

    st.caption("Based on a machine learning model trained on the Iris dataset.")

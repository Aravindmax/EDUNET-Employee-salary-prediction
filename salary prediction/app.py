import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
education_encoder = joblib.load("education_encoder.pkl")
occupation_encoder = joblib.load("occupation_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

# Streamlit setup
st.set_page_config(page_title="Employee Income Classifier", page_icon="💼")
st.title("💼 Income Prediction App")
st.markdown("Predict whether an employee earns **>50K or ≤50K**.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education", education_encoder.classes_)
occupation = st.sidebar.selectbox("Occupation", occupation_encoder.classes_)
gender = st.sidebar.selectbox("Gender", gender_encoder.classes_)
hours = st.sidebar.slider("Hours per Week", 1, 80, 40)

# Preprocess inputs
input_df = pd.DataFrame({
    "age": [age],
    "education": education_encoder.transform([education]),
    "occupation": occupation_encoder.transform([occupation]),
    "gender": gender_encoder.transform([gender]),
    "hours-per-week": [hours]
})

st.write("### 🔍 Input Data")
st.write(input_df)

# Predict button
if st.button("🔮 Predict Income"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch upload
st.markdown("---")
st.subheader("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    batch = pd.read_csv(uploaded_file)

    # Encode categorical fields
    batch["education"] = education_encoder.transform(batch["education"])
    batch["occupation"] = occupation_encoder.transform(batch["occupation"])
    batch["gender"] = gender_encoder.transform(batch["gender"])

    preds = model.predict(batch)
    batch["Predicted Income"] = preds

    st.write("📊 Predictions:")
    st.write(batch.head())

    # Downloadable CSV
    csv = batch.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", csv, file_name="predictions.csv", mime="text/csv")

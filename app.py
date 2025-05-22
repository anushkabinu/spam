import joblib
import streamlit as st

# These files must be in the same directory
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
# streamlit_app.py

import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Spam Email Classifier")

input_text = st.text_area("Enter the email content here:")

if st.button("Classify"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vectorized_input = vectorizer.transform([input_text])
        prediction = model.predict(vectorized_input)[0]
        result = "🚫 Spam" if prediction == 1 else "✅ Ham (Not Spam)"
        st.success(f"Prediction: {result}")

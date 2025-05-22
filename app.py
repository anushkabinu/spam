import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Spam vs Ham Classifier")
st.write("Enter an email or message and classify it as spam or ham.")

# Text input
message = st.text_area("Message Text", height=150)

if st.button("Classify"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]
        label = "Spam" if prediction == 1 else "Ham"
        st.success(f"Predicted: {label}")

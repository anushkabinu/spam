import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message and find out if it's **spam** or **ham**.")

# Text input
user_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Vectorize and predict
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        label = "📨 Ham (Not Spam)" if prediction == 0 else "🚫 Spam"
        st.success(f"Prediction: **{label}**")

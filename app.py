import streamlit as st
import joblib

st.title("🛍️ Fake Review Detection System")
st.write("Detect  whether  a  product  review  is  **Genuine**  or  **Fake**  using  Sentiment  Analysis  and  Machine  Learning.")

# Load model
try:
    model_data = joblib.load("models/model.joblib")
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
except:
    st.error("Model not found! Please run `train.py` first.")
    st.stop()

# Input review
review = st.text_area("Enter a product review:")
if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vec = vectorizer.transform([review])
        pred = model.predict(review_vec)[0]
        proba = model.predict_proba(review_vec).max()

        label = "🟢 Genuine" if pred == "genuine" else "🔴 Fake"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {proba:.2f}")

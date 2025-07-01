# app/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
from utils.preprocess import clean_text
from utils.explain import explain_prediction

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection System")

input_text = st.text_area("Enter a news article or paragraph for classification:")

if st.button("Detect"):
    if input_text.strip():
        cleaned = clean_text(input_text)
        pred = model.predict(vectorizer.transform([cleaned]))[0]
        label = "🟢 Real News" if pred == 0 else "🔴 Fake News"
        st.subheader(f"Prediction: {label}")

        with st.expander("🔍 See Explanation"):
            exp = explain_prediction(cleaned, model, vectorizer)
            st.components.v1.html(exp.as_html(), height=600, scrolling=True)
    else:
        st.warning("Please enter some text first.")

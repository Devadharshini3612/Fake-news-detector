import streamlit as st
from src.predict import test_news

st.title("📰 Fake News Detector")

user_input = st.text_area("Enter a news article or headline to check:")

if st.button("Predict"):
    result = test_news(user_input)
    st.success(f"The article is: **{result}**")

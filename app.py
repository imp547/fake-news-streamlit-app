# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:25:41 2025

@author: mk
"""

import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    # Load data (using try-except for file handling)
    with open("fake_news_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Streamlit app
    st.title("Fake News Detection")
    st.write("Let's predict the news integrity")

    news_input = st.text_area("Put your text here", height=200)

    if st.button("Analyze News"):
        if not news_input.strip():
            st.warning("Please enter news to analyze")
        else:
            try:
                # Strip input before vectorization
                news_input = news_input.strip()
                news_vector = vectorizer.transform([news_input])

                prediction = model.predict(news_vector)[0]

                if prediction == 1:
                    st.success("This news is True")
                else:
                    st.error("It's fake news")

            except Exception as e:  # Catch potential vectorization errors
                st.error(f"An error occurred during analysis: {e}")

except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please ensure they are in the correct directory.")
except Exception as e:
    st.error(f"An error occurred during loading: {e}")



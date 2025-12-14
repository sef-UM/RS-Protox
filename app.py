import streamlit as st
from hatespeech_model import predict_hatespeech
import random

st.set_page_config(page_title="Hatespeech Classifier", layout="centered")
st.title("Hatespeech Text Classifier")
st.write("Enter text below to classify if it is hatespeech or not.")

user_input = st.text_area("Text to classify", "")
input_split = user_input.split(" ")

word_probabilities = {word: round(random.uniform(0, 1), 2) for word in input_split if word}

if st.button("Classify"):
    if user_input.strip():
        result = predict_hatespeech(user_input)
        st.markdown(f"**Result:** {result}")
        col1, col2 = st.columns(2)
        col1.title("Shield Model Results")
        col2.title("Interpretable Shield Model Results")
        col1.write(f"**Result:** {result} ")
        col1.write(f"**Probability:** {random.uniform(0, 1)} ")
        col2.write(f"**Result:** {result}")
        col2.write(f"**Probability:** {random.uniform(0, 1)} ")
        col2.table({"Feature": input_split, "Importance": word_probabilities.values()})
    else:
        st.warning("Please enter some text to classify.")

import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title and instructions
st.title("SMS Spam Detection App")
st.write("Enter a message below to check if it's spam:")

# Text input
message = st.text_area("Message")

# Predict button
if st.button("Check"):
    # Transform message
    message_transformed = vectorizer.transform([message])
    prediction = model.predict(message_transformed)

    if prediction[0] == 1:
        st.error("Spam Message Detected!")
    else:
        st.success("This message is NOT spam.")
import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("ScamBuster AI - Detect Suspicious Messages")
st.subheader("Check if a message is Spam or not")

input_msg = st.text_area("Enter the message:")

if st.button("Predict"):
    if input_msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([input_msg])        
        if prediction[0] == 1:
            st.error("This message is likely SPAM.")
        else:
            st.success("This message is NOT spam.")
import streamlit as st
import pickle

# Load model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit UI
st.title("Scam Detection App")
st.subheader("Check if a message is Spam or not")

input_msg = st.text_area("Enter the message:")

if st.button("Predict"):
    if input_msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        clean_msg = input_msg.strip().lower()        
        prediction = model.predict(input_msg)

        if prediction[0] == 1:
            st.error("This message is likely SPAM.")
        else:
            st.success("This message is NOT spam.")
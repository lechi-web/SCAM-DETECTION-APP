import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from groq import Groq

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Streamlit UI
# Streamlit UI

st.set_page_config(
    page_title="ScamBuster AI",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 ScamBuster AI")
st.write(
    "An AI-powered scam detection assistant that analyzes suspicious messages, "
    "detects scam patterns, and explains potential risks."
)

st.divider()

st.subheader("📩 Enter a suspicious message")

input_msg = st.text_area(
    "Paste the message here:",
    height=150,
    placeholder="Example: Your account has been suspended. Click here to verify..."
)

if st.button("Predict"):
    if input_msg.strip() == "":
        st.warning("Please enter a message.")

    else:
        prediction = model.predict([input_msg])

        if prediction[0] == 1:
            st.error("🚨 High Risk: This message is likely a scam.")
        else:
           st.info("🔍 Scanning message...")


    response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Analyze this message for scam indicators.

                    Message:
                    {input_msg}

                    Explain:
                    1. Is this likely a scam?
                    2. What scam type could it be?
                    3. What are the red flags?
                    4. What should the user do?
                    """
                }
            ]
        )

    analysis = response.choices[0].message.content
    st.success("✅ Scan completed. Results ready.")

    st.subheader("🤖 AI Security Report")

    if "scam" in analysis.lower() or "phishing" in analysis.lower():
            st.error("🚨 AI Assessment: This message is likely SPAM.")
    else:
            st.success("✅ AI Assessment: No strong scam indicators detected.")

    st.write(analysis)
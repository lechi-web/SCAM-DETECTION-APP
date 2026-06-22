import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(dotenv_path=".env")

print("Current folder:", os.getcwd())
print("API KEY FOUND:", os.getenv("GROQ_API_KEY"))

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Analyze this message for scam indicators: Congratulations! You won ₦500,000. Click here now."
        }
    ]
)

print(response.choices[0].message.content)
from groq import Groq

client = Groq(
    api_key="YOUR_API_KEY"
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
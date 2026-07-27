import pandas as pd
from dotenv import load_dotenv
import os
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt
prompt = """
Generate 20 realistic banking phishing SMS messages.

Rules:
- They must look like real SMS messages.
- One message per line.
- Do not number them.
- Do not add bullet points.
- Do not add explanations.
- Return ONLY the messages.
"""

# Send request to Groq
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Get generated text
generated_text = response.choices[0].message.content

# Convert each line into a list
messages = []

for line in generated_text.split("\n"):
    line = line.strip()

    if line:
        messages.append(["spam", line])

# Create DataFrame
df = pd.DataFrame(messages, columns=["v1", "v2"])

# Save CSV
df.to_csv("generated_phishing_messages.csv", index=False)

print("✅ Dataset generated successfully!")
print(df.head())
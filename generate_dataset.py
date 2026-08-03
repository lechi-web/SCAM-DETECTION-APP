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
Generate 30 realistic phishing SMS messages commonly seen in Nigeria.

Requirements:

- Use Nigerian banks and fintech companies.
- Include:
    - GTBank
    - Access Bank
    - Zenith Bank
    - UBA
    - First Bank
    - Fidelity Bank
    - Opay
    - PalmPay
    - Moniepoint
    - Kuda

Common scam themes:

- BVN verification
- NIN update
- KYC verification
- Debit card blocked
- Suspicious login
- Fake transfers
- Fake debit alerts
- Account suspension
- Loan offers
- POS settlement
- OTP requests

Rules:

- Each message should sound realistic.
- Use urgency naturally.
- Include fake links where appropriate.
- One message per line.
- No numbering.
- No explanations.
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
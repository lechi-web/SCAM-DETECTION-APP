import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from utils.text_cleaner import clean_message

# ==========================================
# Load environment variables
# ==========================================
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ==========================================
# Welcome Screen
# ==========================================

print("=" * 40)
print("      ScamBuster Dataset Engine")
print("=" * 40)

print("\nAvailable Categories\n")

print("1. 🏦 Banking Scams")
print("2. 💼 Job Scams")
print("3. 💰 Investment Scams")
print("4. 📦 Delivery Scams")

print("\n" + "-" * 40)

# ==========================================
# User Input
# ==========================================

choice = input("\nEnter your choice (1-4): ")

prompt_files = {
    "1": "prompts/banking_prompt.txt",
    "2": "prompts/job_prompt.txt",
    "3": "prompts/investment_prompt.txt",
    "4": "prompts/delivery_prompt.txt"
}

dataset_names = {
    "1": "banking_scams.csv",
    "2": "job_scams.csv",
    "3": "investment_scams.csv",
    "4": "delivery_scams.csv"
}

if choice not in prompt_files:
    print("❌ Invalid choice.")
    exit()

count = input("\nEnter the number of phishing samples to generate: ")

if not count.isdigit():
    print("❌ Please enter a valid number.")
    exit()

count = int(count)

if count <= 0:
    print("❌ Number must be greater than zero.")
    exit()

# ==========================================
# Load Prompt
# ==========================================

print("\n📂 Loading prompt...")

with open(prompt_files[choice], "r", encoding="utf-8") as file:
    prompt = file.read()

prompt = prompt.replace("{count}", str(count))

print("✅ Prompt loaded successfully!")

# ==========================================
# Generate Dataset
# ==========================================

print("\n🤖 Contacting Groq AI...")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("✅ AI response received!")

generated_text = response.choices[0].message.content

# Uncomment these if you want to debug later
# print(prompt)
# print(generated_text)

# ==========================================
# Process Messages
# ==========================================

print("\n📝 Processing generated messages...")

messages = []

for line in generated_text.split("\n"):

    line = clean_message(line)

    if line:
        messages.append(["spam", line])

# Keep only the requested number
messages = messages[:count]

# Remove duplicates
unique_messages = []
seen = set()

for label, message in messages:

    if message not in seen:
        seen.add(message)
        unique_messages.append([label, message])

messages = unique_messages

print(f"✅ {len(messages)} messages processed!")

# ==========================================
# Save Dataset
# ==========================================

df = pd.DataFrame(messages, columns=["v1", "v2"])

output_file = os.path.join(
    "datasets",
    dataset_names[choice]
)

print("\n💾 Saving dataset...")

df.to_csv(output_file, index=False)

print("✅ Dataset saved successfully!")

# ==========================================
# Summary
# ==========================================

print("\n" + "=" * 40)
print("      GENERATION SUMMARY")
print("=" * 40)

print(f"Category : {dataset_names[choice]}")
print(f"Messages : {len(df)}")
print(f"Output   : {output_file}")

print("=" * 40)
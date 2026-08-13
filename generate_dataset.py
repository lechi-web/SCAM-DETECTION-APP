import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from utils.text_cleaner import clean_message


# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found.")
    exit()

client = Groq(api_key=api_key)


# ==========================================
# Welcome Screen
# ==========================================

print("=" * 50)
print("          ScamBuster Dataset Engine")
print("=" * 50)

print("\nAvailable Categories\n")

print("1. 🏦 Banking Scams")
print("2. 💼 Job Scams")
print("3. 💰 Investment Scams")
print("4. 📦 Delivery Scams")

print("\n" + "-" * 50)


# ==========================================
# Dataset Configuration
# ==========================================

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


# ==========================================
# User Input
# ==========================================

choice = input("\nEnter your choice (1-4): ")

if choice not in prompt_files:
    print("❌ Invalid choice.")
    exit()


count = input(
    "\nEnter the number of phishing samples to generate: "
)

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

with open(
    prompt_files[choice],
    "r",
    encoding="utf-8"
) as file:

    prompt_template = file.read()


# Replace {count}

prompt = prompt_template.replace(
    "{count}",
    str(count)
)


print("✅ Prompt loaded successfully!")


# ==========================================
# Generate Messages
# ==========================================

print("\n🤖 Contacting Groq AI...")
print(f"🎯 Requested messages: {count}")

response = client.chat.completions.create(

    model="llama-3.3-70b-versatile",

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],

    temperature=0.8,

    max_tokens=8000
)


print("✅ AI response received!")


# ==========================================
# Get AI Response
# ==========================================

generated_text = (
    response
    .choices[0]
    .message
    .content
)


# ==========================================
# Process Messages
# ==========================================

print("\n📝 Processing generated messages...")

messages = []

seen = set()


for line in generated_text.splitlines():

    line = clean_message(line)

    if not line:
        continue

    # Remove numbering such as:
    # 1.
    # 2)
    # 3 -
    if line[:2].strip(".-)").isdigit():
        line = line[2:].strip()

    # Remove duplicate messages
    normalized = line.lower().strip()

    if normalized in seen:
        continue

    seen.add(normalized)

    messages.append([
        "spam",
        line
    ])


# Keep requested amount
messages = messages[:count]


# ==========================================
# Check Result
# ==========================================

print(
    f"✅ {len(messages)} unique messages processed!"
)


if len(messages) < count:

    print(
        f"⚠️ Warning: AI returned only "
        f"{len(messages)} usable messages "
        f"out of {count} requested."
    )

    print(
        "💡 You can run the generator again "
        "to create more samples."
    )


# ==========================================
# Create DataFrame
# ==========================================

df = pd.DataFrame(
    messages,
    columns=["v1", "v2"]
)


# ==========================================
# Make Sure Dataset Folder Exists
# ==========================================

os.makedirs(
    "datasets",
    exist_ok=True
)


# ==========================================
# Save Dataset
# ==========================================

output_file = os.path.join(
    "datasets",
    dataset_names[choice]
)


print("\n💾 Saving dataset...")

df.to_csv(
    output_file,
    index=False
)


print(
    f"✅ Dataset saved successfully!"
)


# ==========================================
# Summary
# ==========================================

print("\n" + "=" * 50)
print("             GENERATION SUMMARY")
print("=" * 50)

print(
    f"Category : {dataset_names[choice]}"
)

print(
    f"Requested: {count}"
)

print(
    f"Generated: {len(df)}"
)

print(
    f"Output   : {output_file}"
)

print("=" * 50)


# ==========================================
# Preview
# ==========================================

if not df.empty:

    print("\n📋 Dataset Preview:\n")

    print(
        df.head(10).to_string(index=False)
    )

else:

    print(
        "\n❌ No messages were generated."
    )
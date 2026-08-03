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
print("\nSelect dataset to generate:\n")

print("1. Banking")
print("2. Job")
print("3. Investment")
print("4. Delivery")

choice = input("\nEnter your choice (1-4): ")

prompt_files = {
    "1": "prompts/banking_prompt.txt",
    "2": "prompts/job_prompt.txt",
    "3": "prompts/investment_prompt.txt",
    "4": "prompts/delivery_prompt.txt"
}

if choice not in prompt_files:
    print("❌ Invalid choice.")
    exit()

with open(prompt_files[choice], "r", encoding="utf-8") as file:
    prompt = file.read()

print("\n========== PROMPT BEING SENT ==========\n")
print(prompt)
print("\n=======================================\n")

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
dataset_names = {
    "1": "banking_scams.csv",
    "2": "job_scams.csv",
    "3": "investment_scams.csv",
    "4": "delivery_scams.csv"
}

output_file = os.path.join("datasets", dataset_names[choice])

df.to_csv(output_file, index=False)

print(f"✅ Dataset saved successfully as {output_file}")
print(df.head())
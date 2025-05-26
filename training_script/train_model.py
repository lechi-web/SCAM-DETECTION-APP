# train_model.py

import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Map labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' link ', text)
    # Replace links
    text = re.sub(r'\d+', ' number ', text)
    # Replace numbers
    text = re.sub(r'[^\w\s]', '', text)
    # Remove punctuation
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

#TF-IDF + Naive Bayes pipeline
pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('nb', MultinomialNB())])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate accuracy
y_pred = pipeline.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as model_file:
    pickle.dump(pipeline, model_file)

print("Model saved as model.pkl")
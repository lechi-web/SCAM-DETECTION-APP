# train_model.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load and clean data
df = pd.read_csv("spam.csv", encoding="latin1")[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Customized TF-IDF + Naive Bayes pipeline
pipeline = Pipeline([('tfidf', TfidfVectorizer(
    lowercase=True, stop_words='english', 
    ngram_range=(1, 2), token_patter=r'\b\w+\b'
    )),
    ('nb', MultinomialNB())])

# Train pipeline
pipeline.fit(df['message'], df['label'])

# Save single model pipeline
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Smart pipeline saved as model.pkl")
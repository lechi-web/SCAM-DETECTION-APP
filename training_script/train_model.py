# train_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load and clean data
df = pd.read_csv("datasets/spam.csv", encoding="latin1")[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# Customized TF-IDF + Naive Bayes pipeline
pipeline = Pipeline([('tfidf', TfidfVectorizer(
    lowercase=True, stop_words='english', 
    ngram_range=(1, 2), token_pattern=r'\b\w+\b'
    )),
    ('nb', MultinomialNB())])

# Train pipeline
pipeline.fit(X_train, y_train)

# Test Model
predictions = pipeline.predict(X_test)

# Evaluation metrics
print("Accuracy :", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall   :", recall_score(y_test, predictions))
print("F1 Score :", f1_score(y_test, predictions))

print("\nClassification Report")
print(classification_report(y_test, predictions))

# Save single model pipeline
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Smart pipeline saved as model.pkl")
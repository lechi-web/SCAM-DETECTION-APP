import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ==========================================
# ScamBuster AI - Model Training Engine
# ==========================================

print("=" * 50)
print("       ScamBuster AI - Model Training")
print("=" * 50)


# ==========================================
# Dataset Files
# ==========================================

dataset_files = [
    "datasets/spam.csv",
    "datasets/banking_scams.csv",
    "datasets/job_scams.csv",
    "datasets/investment_scams.csv",
    "datasets/delivery_scams.csv"
]


# ==========================================
# Load Datasets
# ==========================================

print("\n📂 Loading datasets...")

dataframes = []

for file in dataset_files:

    if os.path.exists(file):

        print(f"✅ Loading: {file}")

        try:

            df = pd.read_csv(
                file,
                encoding="latin1"
            )

            if "v1" in df.columns and "v2" in df.columns:

                df = df[["v1", "v2"]]

                df.columns = [
                    "label",
                    "message"
                ]

                dataframes.append(df)

            else:

                print(
                    f"⚠️ Skipping {file}: "
                    "missing v1/v2 columns."
                )

        except Exception as e:

            print(
                f"⚠️ Could not load {file}: {e}"
            )

    else:

        print(
            f"⚠️ Dataset not found: {file}"
        )


# ==========================================
# Combine Datasets
# ==========================================

if not dataframes:

    print(
        "\n❌ No datasets were found."
    )

    exit()


print("\n🔗 Combining datasets...")

df = pd.concat(
    dataframes,
    ignore_index=True
)


# ==========================================
# Clean Dataset
# ==========================================

print("\n🧹 Cleaning dataset...")

df["message"] = (
    df["message"]
    .astype(str)
    .str.strip()
)

df["label"] = (
    df["label"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ==========================================
# Convert Labels
# ==========================================

# ham  = legitimate
# spam = scam

df["label"] = df["label"].map(
    {
        "ham": 0,
        "spam": 1
    }
)


# Remove invalid rows

df = df.dropna(
    subset=["label", "message"]
)


# ==========================================
# Remove Empty Messages
# ==========================================

df = df[
    df["message"].str.len() > 0
]


# ==========================================
# Remove Duplicate Messages
# ==========================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["message"]
)

after_duplicates = len(df)

print(
    f"🧹 Removed "
    f"{before_duplicates - after_duplicates} "
    "duplicate messages."
)


# ==========================================
# Dataset Summary
# ==========================================

print("\n" + "=" * 50)
print("             DATASET SUMMARY")
print("=" * 50)

print(
    f"Total messages: {len(df)}"
)

print(
    f"Legitimate messages: "
    f"{sum(df['label'] == 0)}"
)

print(
    f"Scam messages: "
    f"{sum(df['label'] == 1)}"
)

print("=" * 50)


# ==========================================
# Features and Labels
# ==========================================

X = df["message"]
y = df["label"]


# ==========================================
# Train/Test Split
# ==========================================

print("\n✂️ Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ==========================================
# TF-IDF + Logistic Regression
# ==========================================

print(
    "\n🧠 Building ScamBuster ML pipeline..."
)

print(
    "🔧 Engine: TF-IDF + Logistic Regression"
)

pipeline = Pipeline(

    [

        (
            "tfidf",

            TfidfVectorizer(

                lowercase=True,

                stop_words=None,

                ngram_range=(1, 2),

                token_pattern=r"(?u)\b\w+\b",

                min_df=1,

                max_features=50000
            )
        ),

        (
            "classifier",

            LogisticRegression(

                max_iter=1000,

                class_weight="balanced",

                random_state=42
            )
        )

    ]
)


# ==========================================
# Train Model
# ==========================================

print(
    "\n🚀 Training ScamBuster model..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "✅ Model training completed!"
)


# ==========================================
# Test Model
# ==========================================

print(
    "\n🔍 Testing model..."
)

predictions = pipeline.predict(
    X_test
)


# ==========================================
# Evaluation Metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# ==========================================
# Display Metrics
# ==========================================

print("\n" + "=" * 50)
print("             MODEL PERFORMANCE")
print("=" * 50)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ==========================================
# Classification Report
# ==========================================

print(
    "\n📊 Classification Report"
)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Scam"
        ],
        zero_division=0
    )
)


# ==========================================
# Confusion Matrix
# ==========================================

print(
    "📊 Confusion Matrix"
)

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)


# ==========================================
# Save Model
# ==========================================

print(
    "\n💾 Saving trained model..."
)

with open(
    "model.pkl",
    "wb"
) as f:

    pickle.dump(
        pipeline,
        f
    )

print(
    "✅ ScamBuster model saved as model.pkl"
)


# ==========================================
# Final Summary
# ==========================================

print("\n" + "=" * 50)
print("        TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)

print(
    f"📚 Total training data: {len(df)} messages"
)

print(
    f"🎯 Accuracy: {accuracy:.2%}"
)

print(
    f"🎯 Precision: {precision:.2%}"
)

print(
    f"🎯 Recall: {recall:.2%}"
)

print(
    f"🎯 F1 Score: {f1:.2%}"
)

print(
    "\n🛡️ ScamBuster AI model is ready."
)

print("=" * 50)
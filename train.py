import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w.isalpha() and w not in stop_words]
    return " ".join(words)

def main():
    os.makedirs("models", exist_ok=True)

    # Load dataset
    df = pd.read_csv("data/reviews_sample.csv")
    df.dropna(subset=['review_text', 'label'], inplace=True)

    # Preprocess text
    df["clean_text"] = df["review_text"].astype(str).apply(preprocess_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42
    )

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Evaluate
    preds = model.predict(X_test_vec)
    print("=== Classification Report ===")
    print(classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))

    # Save model and vectorizer
    joblib.dump({"model": model, "vectorizer": vectorizer}, "models/model.joblib")
    print("\n✅ Model saved successfully at models/model.joblib")

if __name__ == "__main__":
    main()

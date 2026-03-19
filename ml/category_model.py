import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def train_nlp_classifier(df):
    """
    Trains a real Machine Learning pipeline to categorize transactions based on their descriptions.
    """
    print("\n" + "="*50)
    print("TRAINING NLP CATEGORIZATION PIPELINE...")
    print("="*50)

    # 1. Clean the data (Drop rows where Description or Category is missing)
   
    df_clean = df.dropna(subset=['Description', 'Category']).copy()

    if len(df_clean) < 10:
        print("Not enough categorized data to train. Need at least 10 rows.")
        return None

    # 2. Define our Features (X) and Target Labels (y)
    X = df_clean['Description']
    y = df_clean['Category']

    # 3. PROPER ML PIPELINE: The Train/Test Split
    # We keep 80% to train, and lock away 20% for the final exam.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Build the AI Architecture
    # TfidfVectorizer: Converts text words into mathematical weights
    # MultinomialNB: A classic algorithm perfect for text classification
    nlp_model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

    # 5. Train the Model (The student studies the textbook)
    nlp_model.fit(X_train, y_train)

    # 6. Evaluate the Model (The student takes the final exam)
    predictions = nlp_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Print the professional pipeline metrics
    print(f"Training Data Size : {len(X_train)} transactions")
    print(f"Hidden Test Size   : {len(X_test)} transactions")
    print(f"Model Accuracy     : {accuracy * 100:.1f}%")
    print("="*50 + "\n")

    return nlp_model

import joblib
import os

def predict_category(description):
    model_path = 'ml/saved_brain.pkl'
    
    # Check if the brain exists
    if not os.path.exists(model_path):
        return "Uncategorized (Please run ml/train_model.py first!)"
    
    # Load the brain instantly
    model = joblib.load(model_path)
    
    # Make the prediction
    prediction = model.predict([description])[0]
    return prediction

# Test it out!
if __name__ == "__main__":
    test_desc = "UBER RIDES SF #8392"
    print(f"'{test_desc}' was categorized as: {predict_category(test_desc)}")
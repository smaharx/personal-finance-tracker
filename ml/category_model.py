from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def train_classifier(category_keywords):
    """Trains a Naive Bayes NLP model to categorize transactions."""
    print("\n" + "="*45)
    print("BOOTING AI CATEGORIZATION ENGINE...")
    
    training_texts = []
    training_labels = []
    
    # Safety Check: If memory is totally empty, give it a placeholder
    if not category_keywords:
        category_keywords = {'Other': ['unknown']}
        
    for category, keywords in category_keywords.items():
        for word in keywords:
            training_texts.append(word)
            training_labels.append(category)
            
    # Build and train the Machine Learning Pipeline
    try:
        classifier = make_pipeline(CountVectorizer(), MultinomialNB())
        classifier.fit(training_texts, training_labels)
        print("AI Brain is online and ready to categorize!")
    except Exception as e:
        print(f"Critical Error training AI: {e}")
        return None
        
    print("="*45 + "\n")
    return classifier

def predict_category(text, classifier):
    """Uses the trained AI model to guess the category of a new transaction."""
    if classifier is None or not isinstance(text, str) or not text.strip():
        return 'Other'
        
    try:
        prediction = classifier.predict([text.lower()])
        return prediction[0]
    except:
        return 'Other'
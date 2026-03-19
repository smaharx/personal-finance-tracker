import joblib
import os

def predict_category(description):
    model_path = 'ml/saved_brain.pkl'
    
    # Safety check: Make sure the brain exists!
    if not os.path.exists(model_path):
        return "Uncategorized (No AI model found)"
    
    # Load the brain
    model = joblib.load(model_path)
    
    # Make the prediction
    prediction = model.predict([description])[0]
    
    return prediction

# Quick test to make sure it works!
if __name__ == "__main__":
    test_1 = "UBER RIDES SF"
    test_2 = "AMZN MKTP US #9923"
    test_3 = "CITY APARTMENTS LEASING"
    
    print(f"'{test_1}' -> {predict_category(test_1)}")
    print(f"'{test_2}' -> {predict_category(test_2)}")
    print(f"'{test_3}' -> {predict_category(test_3)}")
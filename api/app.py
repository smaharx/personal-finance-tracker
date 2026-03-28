from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Finance AI Inference API")

# Load the model once when the server starts
MODEL_PATH = os.path.join("ml", "saved_brain.pkl")
model = joblib.load(MODEL_PATH)

class Transaction(BaseModel):
    description: str

@app.post("/predict")
def predict(item: Transaction):
    # The actual AI logic
    prediction = model.predict([item.description])[0]
    return {"description": item.description, "category": prediction}

@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": True}
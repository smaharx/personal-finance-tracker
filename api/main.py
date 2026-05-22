import os
import joblib
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Import our custom database configurations and models
from api.database import get_db
from api.models import TransactionModel

# Initialize the API application
app = FastAPI(title="Finance Tracker API", version="2.0")

# Define the data structure we expect from the frontend for AI classification
class TransactionCreate(BaseModel):
    description: str

# Safely load the ML model
MODEL_PATH = os.path.join("ml", "saved_brain.pkl")
try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except FileNotFoundError:
    model = None
    MODEL_LOADED = False


# ==========================================
# ENDPOINT 1: SERVICE HEALTH CHECK
# ==========================================
@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "FastAPI backend is running successfully.",
        "version": "2.0",
        "ai_model_loaded": MODEL_LOADED
    }


# ==========================================
# ENDPOINT 2: AI CATEGORY INFERENCE
# ==========================================
@app.post("/predict")
def predict_category(item: TransactionCreate):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="AI Model is not loaded. Train the model first.")
    
    prediction = model.predict([item.description])[0]
    return {
        "description": item.description, 
        "predicted_category": prediction
    }


# ==========================================
# ENDPOINT 3: REFRACTORED ORM DATA FETCHING
# ==========================================
@app.get("/transactions")
def get_transactions(limit: int = 50, db: Session = Depends(get_db)):
    """
    Fetches the most recent transactions using SQLAlchemy ORM expressions.
    Injects the database session using FastAPI's dependency injection system.
    """
    try:
        # We query the database using the Python Class instead of hardcoded SQL strings
        transactions = db.query(TransactionModel).order_by(TransactionModel.date.desc()).limit(limit).all()
        
        # Serialize the SQLAlchemy objects into a clean JSON structure
        serialized_transactions = [
            {
                "id": t.id,
                "date": t.date,
                "description": t.description,
                "category": t.category,
                "amount": t.amount,
                "is_anomaly": t.is_anomaly
            }
            for t in transactions
        ]
        
        return {"count": len(serialized_transactions), "transactions": serialized_transactions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database abstraction layer error: {str(e)}")
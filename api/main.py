import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3

# Initialize the API application
app = FastAPI(title="Finance Tracker API", version="2.0")

# Define the data structure we expect from the frontend
class Transaction(BaseModel):
    description: str

# Safely load the ML model
MODEL_PATH = os.path.join("ml", "saved_brain.pkl")
try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except FileNotFoundError:
    model = None
    MODEL_LOADED = False

# 1. Health Check Endpoint
@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "FastAPI backend is running successfully.",
        "version": "2.0",
        "ai_model_loaded": MODEL_LOADED
    }

# 2. AI Prediction Endpoint (Uncommented and fixed)
@app.post("/predict")
def predict_category(item: Transaction):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="AI Model is not loaded. Train the model first.")
    
    # Run the prediction
    prediction = model.predict([item.description])[0]
    return {
        "description": item.description, 
        "predicted_category": prediction
    }




# Helper function to connect to the database securely
def get_db_connection():
    # Looking for the database in the root folder
    conn = sqlite3.connect("expenses.db")
    # This crucial line tells SQLite to return rows as dictionaries instead of plain tuples
    conn.row_factory = sqlite3.Row 
    return conn

# 3. Fetch Transactions Endpoint
@app.get("/transactions")
def get_transactions(limit: int = 50):
    """
    Fetches the most recent transactions from the database.
    Defaults to 50 records unless a different limit is specified.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # SQL Query to get the latest transactions
        cursor.execute("SELECT * FROM transactions ORDER BY date DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        # Convert the SQLite rows into standard Python dictionaries for JSON output
        return {"count": len(rows), "transactions": [dict(row) for row in rows]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")   
import pandas as pd
import os

def load_csv(filepath, classifier):
    """Loads, cleans, and categorizes transaction data from a CSV file."""
    print(f"Loading data from {filepath}...")
    
    # 1. Check if the file actually exists
    if not os.path.exists(filepath):
        print(f"Error: Could not find '{filepath}'.")
        return None

    # 2. Try loading the CSV
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

    # 3. Check if the file is completely empty
    if data.empty:
        print("The CSV file is empty. Please add some transactions!")
        return None

    # 4. Ensure the required columns exist
    required_columns = ["Date", "Description", "Amount"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Error: Your CSV is missing required columns: {missing_columns}")
        return None

    # 5. Clean and format the Dates
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    initial_rows = len(data)
    data = data.dropna(subset=["Date"])
    
    # 6. Clean the Amounts (Removes $, commas, and converts to float)
    data["Amount"] = data["Amount"].astype(str).str.replace(r'[$,]', '', regex=True)
    data["Amount"] = pd.to_numeric(data["Amount"], errors="coerce")
    data = data.dropna(subset=["Amount"])
    
    dropped_rows = initial_rows - len(data)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to blank or invalid data.")

    # 7. Unleash the AI to categorize descriptions
    if classifier is not None:
        data["Category"] = data["Description"].apply(
            lambda x: classifier.predict([str(x)])[0]
        )
    else:
        data["Category"] = "Other"

    print(f"Successfully loaded and categorized {len(data)} transactions!")
    return data
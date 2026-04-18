import sqlite3
import random
from datetime import datetime, timedelta

# Configuration
NUM_RECORDS = 20000
DB_PATH = 'data/expenses.db'

# Realistic categories and corresponding descriptions
CATEGORIES = {
    "Groceries": ["Walmart", "Whole Foods", "Local Supermarket", "Target", "Trader Joe's"],
    "Transport": ["Uber", "Gas Station", "Subway Pass", "Lyft", "Parking fee"],
    "Utilities": ["Electric Bill", "Water Bill", "Internet Provider", "Phone Bill"],
    "Entertainment": ["Netflix", "Movie Theater", "Spotify", "Steam Games", "Concert Tickets"],
    "Dining": ["Starbucks", "Local Restaurant", "McDonalds", "Pizza Delivery", "Sushi Bar"],
    "Shopping": ["Amazon", "Clothing Store", "Best Buy", "Home Depot"]
}

def generate_mock_data():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if it doesn't exist (safety check)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Date TEXT,
            Description TEXT,
            Amount REAL,
            Category TEXT
        )
    """)

    # Clear existing data to prevent duplicates during testing
    cursor.execute("DELETE FROM transactions")
    
    transactions = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730) # 2 years of history for Prophet

    print(f"Generating {NUM_RECORDS} realistic transactions...")
    
    for _ in range(NUM_RECORDS):
        # 1. Generate random date within the last 2 years
        random_days = random.randrange((end_date - start_date).days)
        txn_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

        # 2. Pick a random category and description
        category = random.choice(list(CATEGORIES.keys()))
        description = random.choice(CATEGORIES[category])

        # 3. Generate realistic amounts based on category
        if category == "Utilities":
            amount = round(random.uniform(50.0, 200.0), 2)
        elif category == "Groceries":
            amount = round(random.uniform(30.0, 300.0), 2)
        elif category == "Entertainment":
            amount = round(random.uniform(9.99, 100.0), 2)
        else:
            amount = round(random.uniform(5.0, 150.0), 2)

        # 4. Inject controlled anomalies for the Isolation Forest to find (~1% chance)
        if random.random() < 0.01:
            amount = round(random.uniform(1000.0, 5000.0), 2)
            description = f"ANOMALY: {description} (Large Purchase)"

        transactions.append((txn_date, description, amount, category))

    # Bulk insert for maximum performance
    print("Injecting data into SQLite...")
    cursor.executemany(
        "INSERT INTO transactions (Date, Description, Amount, Category) VALUES (?, ?, ?, ?)", 
        transactions
    )

    conn.commit()
    conn.close()
    print("✅ Successfully seeded 20,000 records into expenses.db!")

if __name__ == "__main__":
    generate_mock_data()
    
import pandas as pd
import random
from datetime import datetime, timedelta
import os

print("Generating 5,000 localized transactions...")

# 🌟 THE NEW, LOCALIZED AI VOCABULARY 🌟
# We added your specific words, plus common regional brands to make the AI super smart!
CATEGORIES = {
    "Food": ["Starbucks", "McDonalds", "Chai", "Sambosa", "Biryani", "Foodpanda", "KFC", "Cafe", "Restaurant"],
    "Transport": ["Uber Ride", "Careem", "Petrol", "Shell Oil", "Bykea", "Bus Ticket", "InDrive", "PSO Pump"],
    "Shopping": ["Amazon", "Target", "Walmart", "Eid Clothes", "Bazaar", "Daraz", "Imtiaz Super Market", "Grocery"],
    "Utilities": ["Electric Bill", "K-Electric", "Water Bill", "Sui Gas", "PTCL Internet"],
    "Subscriptions": ["Netflix Subscription", "Spotify Premium", "Gym Membership", "Cloud Storage"],
    "Entertainment": ["Cinema", "Nueplex", "Steam Games", "Concert Ticket"],
    "Rent": ["Apartment Rent", "House Leasing", "Hostel Dues"],
    "Health": ["Pharmacy", "Agha Khan Hospital", "Doctor Clinic", "Panadol", "Medical Store"],
    "Travel": ["Emirates Airline", "PIA Ticket", "Hotel Booking", "Train Ticket"]
}

data = []
start_date = datetime.today() - timedelta(days=365)

for _ in range(5000):
    # Pick a random category
    cat = random.choice(list(CATEGORIES.keys()))
    
    # Pick a random word from that category
    desc = random.choice(CATEGORIES[cat])
    
    # Generate a random amount and date
    amount = round(random.uniform(5.0, 500.0), 2)
    random_days = random.randint(0, 365)
    txn_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
    
    data.append([txn_date, desc, amount, cat])

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Save to CSV (Make sure this matches the filename your train_model.py looks for!)
file_path = "data/synthetic_expenses.csv" 
df = pd.DataFrame(data, columns=["Date", "Description", "Amount", "Category"])
df.to_csv(file_path, index=False)

print(f"✅ Successfully generated 5,000 localized rows in {file_path}!")
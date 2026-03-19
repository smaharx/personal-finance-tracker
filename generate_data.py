import csv
import random
from datetime import datetime, timedelta

# Realistic bank transaction descriptions
TEMPLATES = {
    "Food": ["STARBUCKS STORE", "MCDONALDS", "WHOLE FOODS", "TRADER JOES", "DOORDASH", "UBEREATS", "TACO BELL"],
    "Rent": ["CITY APARTMENTS", "VILLAGE GREEN RENT", "OAKWOOD LEASING"],
    "Transport": ["UBER RIDE", "LYFT", "CHEVRON GAS", "SHELL OIL", "MTA SUBWAY", "EXXONMOBIL"],
    "Subscriptions": ["NETFLIX", "SPOTIFY", "AMAZON PRIME", "HULU", "DISNEY PLUS", "GYM MEMBERSHIP"],
    "Entertainment": ["AMC THEATERS", "TICKETMASTER", "STEAM GAMES", "PLAYSTATION NETWORK", "BOWLING ALLEY"],
    "Shopping": ["AMZN MKTP US", "TARGET", "WALMART", "APPLE STORE", "BEST BUY", "H&M"],
    "Income": ["PAYROLL DIRECT DEP", "VENMO CASHOUT", "STRIPE TRANSFER"]
}

def generate_transactions(num_rows=5000):
    start_date = datetime(2022, 1, 1)
    
    with open('data/synthetic_transactions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Description", "Amount", "Category"])
        
        for _ in range(num_rows):
            category = random.choices(
                list(TEMPLATES.keys()), 
                weights=[25, 5, 15, 10, 10, 30, 5], k=1
            )[0]
            
            description = random.choice(TEMPLATES[category]) + " #" + str(random.randint(1000, 9999))
            
            # Generate realistic amounts based on category
            if category == "Income":
                amount = round(random.uniform(2000, 5000), 2)
            elif category == "Rent":
                amount = round(random.uniform(1000, 2500), 2)
            else:
                amount = round(random.uniform(5, 150), 2)
                
            date = start_date + timedelta(days=random.randint(0, 700))
            
            writer.writerow([date.strftime("%Y-%m-%d"), description, amount, category])

if __name__ == "__main__":
    print("Generating 5,000 synthetic bank transactions...")
    generate_transactions(5000)
    print("Success! Data saved to data/synthetic_transactions.csv")
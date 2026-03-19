import pandas as pd
import sqlite3
from ml.forecast_model import train_model, predict_future, check_accuracy
from ml.category_model import train_nlp_classifier, predict_category
from analysis.spending_analysis import spending_breakdown
from analysis.visualization import plot_pie
from analysis.health_monitor import health_check

class FinanceAssistant:

    def __init__(self, budget_limit):
        self.budget_limit = budget_limit
        self.category_budgets = {}
        self.data = None
        self.model = None
        
        # This will now hold our Scikit-Learn NLP Pipeline
        self.classifier = None 

    def set_category_budget(self, category, amount):
        """Sets a specific spending limit for a category."""
        self.category_budgets[category] = amount
        print(f"Budget for '{category}' set to ${amount:,.2f}")

    def load_data(self):
        """Upgraded: Connects to SQL database instead of CSV."""
        print("\nConnecting to SQL Database...")
        self.db_path = 'data/expenses.db' # Save the path
        
        try:
            # 1. Open SQL Connection
            conn = sqlite3.connect(self.db_path)
            
            # 2. Read data using a real SQL Query!
            self.data = pd.read_sql_query("SELECT * FROM transactions", conn)
            conn.close()
            
        except sqlite3.OperationalError:
            print(" Error: Database not found. Did you run migrate_db.py?")
            return False

        # Train the NLP Pipeline on already categorized data
        self.classifier = train_nlp_classifier(self.data)

        # Predict missing categories (Logic stays exactly the same!)
        if self.classifier is not None:
            missing_mask = self.data['Category'].isna() | (self.data['Category'] == '')
            if missing_mask.sum() > 0:
                print(f" AI is auto-categorizing {missing_mask.sum()} new transactions...")
                
                # Predict
                self.data.loc[missing_mask, 'Category'] = self.data.loc[missing_mask, 'Description'].apply(
                    lambda desc: predict_category(self.classifier, desc)
                )
                
                # 3. Save updates back to SQL
                conn = sqlite3.connect(self.db_path)
                self.data.to_sql('transactions', conn, if_exists='replace', index=False)
                conn.close()
                print(" Saved new AI predictions to SQL Database.")
        
        print(f" Successfully loaded {len(self.data)} transactions from Database!")
        return True

    def train_forecast(self):
        self.model, self.monthly_aggregates = train_model(self.data)

    def predict_future(self, months=3):
        predict_future(self.model, self.monthly_aggregates, months)

    def check_prediction_accuracy(self):
        check_accuracy(self.model, self.monthly_aggregates)

    def analyze_spending(self):
        spending_breakdown(self.data, self.category_budgets)

    def show_pie_chart(self):
        plot_pie(self.data)

    def health_check(self):
        # Passes the whole bot to the health monitor
        health_check(self)

    def teach_the_bot(self):
        """Upgraded: Inserts new data directly into the SQL Database."""
        print("\n---  Teach the NLP AI (SQL Mode) ---")
        
        date = input("Enter date (YYYY-MM-DD) [or press Enter for today]: ").strip()
        if not date:
            date = pd.Timestamp.today().strftime('%Y-%m-%d')
            
        desc = input("Enter the description (e.g., 'UBER EATS'): ").strip()
        cat = input("Enter the correct category (e.g., 'Food'): ").strip()
        
        try:
            amount = float(input("Enter the amount (e.g., 25.50): "))
        except ValueError:
            print(" Invalid amount. Cancelling.")
            return

        # 1. Open SQL Connection
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 2. Execute a raw SQL INSERT command (This is how the pros do it!)
        sql_command = "INSERT INTO transactions (Date, Description, Amount, Category) VALUES (?, ?, ?, ?)"
        cursor.execute(sql_command, (date, desc, amount, cat))
        
        # 3. Commit the transaction and close
        conn.commit()
        conn.close()
        
        print(f" SQL INSERT Successful! The AI will study this row on next boot.")
        
        # Force the bot to reload the data from the DB so memory stays fresh
        self.load_data()
import pandas as pd
import sqlite3
from ml.forecast_model import train_model, predict_future, check_accuracy
from ml.category_model import predict_category
from analysis.spending_analysis import spending_breakdown
from analysis.visualization import plot_pie
from analysis.health_monitor import health_check

class FinanceAssistant:

    def __init__(self, budget_limit):
        self.budget_limit = budget_limit
        self.category_budgets = {}
        self.data = None
        self.model = None
        # Deleted self.classifier - we don't need it anymore! The brain is on the hard drive.

    def set_category_budget(self, category, amount):
        """Saves a budget to both RAM and the permanent SQL Database."""
        
        # 1. Save to RAM (so it updates the current session instantly)
        self.category_budgets[category] = amount
        
        # 2. Save to SQL (so it remembers forever)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Data Science Trick: UPSERT (Update or Insert)
        cursor.execute('''
            INSERT OR REPLACE INTO budgets (Category, Monthly_Limit) 
            VALUES (?, ?)
        ''', (category, amount))
        
        conn.commit()
        conn.close()
        
        print(f" Permanent Budget for '{category}' securely saved to Database as ${amount:,.2f}")

    def load_data(self):
        """Upgraded: Connects to SQL, auto-creates tables, and loads budgets."""
        print("\nConnecting to SQL Database...")
        self.db_path = 'data/expenses.db' 
        
        try:
            # 1. Open SQL Connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 2. CREATE TABLES FIRST (Before we try to read anything)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS budgets (
                    Category TEXT PRIMARY KEY,
                    Monthly_Limit REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Date TEXT,
                    Description TEXT,
                    Amount REAL,
                    Category TEXT
                )
            ''')
            conn.commit()

            # 3. LOAD BUDGETS
            cursor.execute("SELECT Category, Monthly_Limit FROM budgets")
            for row in cursor.fetchall():
                category_name = row[0]
                budget_amount = row[1]
                self.category_budgets[category_name] = budget_amount

            # 4. LOAD TRANSACTIONS
            self.data = pd.read_sql_query("SELECT * FROM transactions", conn)
            conn.close()
            
        except sqlite3.OperationalError as e:
            print(f" Error: Database issue - {e}")
            return False

        # --- THE AI UPGRADE: No more slow training! Just predict instantly. ---
        # Note: If database is completely empty, self.data will be empty, so we skip this
        if not self.data.empty:
            missing_mask = self.data['Category'].isna() | (self.data['Category'] == '')
            if missing_mask.sum() > 0:
                print(f" AI is auto-categorizing {missing_mask.sum()} new transactions...")
                
                # Call the API Brain directly!
                self.data.loc[missing_mask, 'Category'] = self.data.loc[missing_mask, 'Description'].apply(
                    lambda desc: predict_category(desc)
                )
                
                # Save updates back to SQL
                conn = sqlite3.connect(self.db_path)
                self.data.to_sql('transactions', conn, if_exists='replace', index=False)
                conn.close()
        
        # Count how many budgets we loaded to show the user
        budget_count = len(self.category_budgets)
        print(f" Loaded {len(self.data)} transactions and {budget_count} permanent budgets!")
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
        print("\n---  Add Manual Transaction (SQL Mode) ---")
        
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
        
        print(f" SQL INSERT Successful! Row saved to database.")
        
        # Force the bot to reload the data from the DB so memory stays fresh
        self.load_data()
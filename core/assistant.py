import pandas as pd
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

    def load_data(self, filepath):
        """Loads data, trains the NLP model, and auto-categorizes missing rows."""
        print("\nLoading data from CSV...")
        try:
            self.data = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Error: Could not find {filepath}")
            return False

        # Ensure Category column exists
        if 'Category' not in self.data.columns:
            self.data['Category'] = None

        # 1. Train the NLP Pipeline on already categorized data
        self.classifier = train_nlp_classifier(self.data)

        # 2. Predict categories for rows that are missing them
        if self.classifier is not None:
            # Find rows where Category is missing or NaN
            missing_mask = self.data['Category'].isna() | (self.data['Category'] == '')
            
            if missing_mask.sum() > 0:
                print(f"AI is auto-categorizing {missing_mask.sum()} new transactions...")
                # Apply the prediction model to the descriptions of missing rows
                self.data.loc[missing_mask, 'Category'] = self.data.loc[missing_mask, 'Description'].apply(
                    lambda desc: predict_category(self.classifier, desc)
                )
                
                # Save the newly guessed categories back to the CSV permanently
                self.data.to_csv(filepath, index=False)
                print("Saved new AI predictions to CSV.")
        
        print(f"Successfully loaded {len(self.data)} transactions!")
        return True

    def train_forecast(self):
        self.model, self.monthly_aggregates = train_model(self.data)

    def predict_future(self, months=3):
        predict_future(self.model, self.monthly_aggregates, months)

    def check_prediction_accuracy(self):
        check_accuracy(self.model, self.monthly_aggregates)

    def analyze_spending(self):
        spending_breakdown(self.data)

    def show_pie_chart(self):
        plot_pie(self.data)

    def health_check(self):
        # Passes the whole bot to the health monitor
        health_check(self)

    def teach_the_bot(self):
        """Upgraded: Adds a new labeled transaction directly to the CSV training data."""
        print("\n--- Teach the NLP AI ---")
        print("To make the AI smarter, provide an example transaction!")
        
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

        # Create a new row
        new_row = pd.DataFrame([{'Date': date, 'Description': desc, 'Amount': amount, 'Category': cat}])
        
        # Append to our active dataframe
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        
        # Save to CSV so it becomes permanent "textbook" material for the next boot
        self.data.to_csv('data/my_expenses.csv', index=False)
        print(f" Added! The AI will study this new data point the next time you boot up.")
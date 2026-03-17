from core.data_loader import load_csv
from core.memory_manager import load_memory, save_memory

from ml.forecast_model import train_model, predict_future, check_accuracy
from ml.category_model import train_classifier, predict_category

from analysis.spending_analysis import spending_breakdown
from analysis.visualization import plot_pie
from analysis.health_monitor import health_check


class FinanceAssistant:

    def __init__(self, budget_limit):

        self.budget_limit = budget_limit
        self.category_budgets = {}
        self.data = None
        self.model = None
        self.classifier = None

        self.category_keywords = load_memory()

        self.classifier = train_classifier(self.category_keywords)


    def set_category_budget(self, category, amount):
        """Sets a specific spending limit for a category."""
        self.category_budgets[category] = amount
        print(f"Budget for '{category}' set to ${amount:,.2f}")

    def load_data(self, filepath):

        self.data = load_csv(filepath, self.classifier)

        return self.data is not None


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

        health_check(self)


        
    def teach_the_bot(self):
        print("\n---Teach the AI ---")
        category = input("Enter the category (e.g., Food, Transport, Rent): ").strip()
        keyword = input(f"Enter a keyword for {category} (e.g., walmart, uber): ").strip().lower()

        if not category or not keyword:
            print("Input cannot be empty. Skipping.")
            return

        # Create the category if it doesn't exist yet
        if category not in self.category_keywords:
            self.category_keywords[category] = []

        # Add the keyword to the AI's memory
        if keyword not in self.category_keywords[category]:
            self.category_keywords[category].append(keyword)
            
            # Save it to the JSON file using your memory manager
            save_memory(self.category_keywords)
            print(f"Got it! I will now remember that '{keyword}' belongs to {category}.")
            
            # Re-train the AI immediately so it gets smarter right now
            self.classifier = train_classifier(self.category_keywords)
        else:
            print(f"I already know that '{keyword}' belongs to {category}!")    
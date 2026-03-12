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
        self.data = None
        self.model = None
        self.classifier = None

        self.category_keywords = load_memory()

        self.classifier = train_classifier(self.category_keywords)


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

        health_check(self.data, self.budget_limit, self.monthly_aggregates)
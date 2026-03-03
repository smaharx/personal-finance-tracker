import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression #ML
from datetime import datetime  #time matching 
import os  #file checking
import matplotlib.pyplot as plt # Graphs
from sklearn.feature_extraction.text import CountVectorizer #NLP-converts text into numbers
from sklearn.naive_bayes import MultinomialNB  # NLP-categorization 
from sklearn.pipeline import make_pipeline # merging multiple operations into single one
from sklearn.metrics import mean_absolute_error, mean_squared_error  # checking the probability of the error
import json # Files to store the data in the form of dictionary-- 
"""here json is mainly used for storing the data that chatbots learned during what i
gave the data it"""

class FinanceAssistant:
    def __init__(self, budget_limit=50000):
        self.budget_limit = budget_limit
        self.data = None
        self.model = None
        self.classifier = None 
        
        # Load memory from JSON instead of hardcoding it!
        self.load_knowledge()

    def load_knowledge(self):
        """Loads the AI's memory from a JSON file. If none exists or it's empty, creates a default one."""
        self.memory_file = 'ai_memory.json'
        
        # We define the starter pack here so we can use it if things go wrong
        default_memory = {
            'Food': ['mcdonalds', 'burger', 'pizza', 'starbucks', 'cafe', 'foodpanda', 'kfc', 'biryani'],
            'Transport': ['uber', 'gas', 'shell', 'parking', 'careem', 'indrive', 'bykea'],
            'Utilities': ['electric', 'water', 'internet', 'netflix', 'ptcl', 'k-electric', 'sui gas'],
            'Housing': ['rent', 'leasing', 'mortgage', 'apartments'],
            'Shopping': ['daraz', 'clothing', 'amazon', 'mall', 'khaadi', 'gul ahmed']
        }

        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as file:
                    self.category_keywords = json.load(file)
                print("AI Memory loaded successfully.")
            except json.JSONDecodeError:
                # FIX: If the file exists but is empty/corrupted, it catches the error here!
                print("Memory file was empty or corrupted. Rebuilding from scratch...")
                self.category_keywords = default_memory
                self.save_knowledge() # This immediately overwrites the empty file with good data
        else:
            print("No memory found. Starting with default knowledge base.")
            self.category_keywords = default_memory
            self.save_knowledge()

    def save_knowledge(self):
        """Saves the current dictionary to a JSON file so it doesn't forget."""
        with open(self.memory_file, 'w') as file:
            json.dump(self.category_keywords, file, indent=4)    

  
    def check_prediction_accuracy(self):
        """Check how accurate the forecast model is on past data."""
    
        if self.model is None:
            print("Train the model first before checking accuracy.")
            return

        if not hasattr(self, 'monthly_aggregates'):
            print("No monthly data available to check accuracy.")
            return

        # Use the historical 'Month_Index' as input
        X = self.monthly_aggregates[['Month_Index']]
        y_real = self.monthly_aggregates['Amount']

        #  Predict using the model
        y_pred = self.model.predict(X)

        #  Calculate metrics
        mae = mean_absolute_error(y_real, y_pred)
        mse = mean_squared_error(y_real, y_pred)
        rmse = np.sqrt(mse)

        #  Print nicely
        print("\nModel Accuracy Report (on historical data):")
        print(f"Average Error (MAE)    : ${mae:.2f}  → average off by this much")
        print(f"Mean Squared Error (MSE): ${mse:.2f}  → squares big mistakes")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}  → typical error per month")
    
        # Optional: show % error
        mean_actual = y_real.mean()
        pct_error = (mae / mean_actual) * 100
        print(f"Approximate % Error     : {pct_error:.1f}% of average spending")  


        

    def _get_category(self, description):
        """Uses the AI model to guess the category."""
        # Safety check: If model isn't trained, use 'Other'
        if self.classifier is None:
            return 'Other'
            
        # The model expects a list, so we wrap the description in []
        prediction = self.classifier.predict([str(description)])
        return prediction[0]
    
    def load_csv(self, filepath):
        """Loads bulk data from a CSV file and normalizes currency."""
        if not os.path.exists(filepath):
            print(f"Error: The file '{filepath}' was not found.")
            return False

         # --- 1. Load the new CSV ---
        new_data = pd.read_csv(filepath)

         # --- 2. Convert 'Date' to datetime objects ---
        new_data['Date'] = pd.to_datetime(new_data['Date'], errors='coerce')

         # --- 3. Remove rows with invalid dates ---
        new_data = new_data.dropna(subset=['Date'])

        # --- 4. Auto-Categorize Transactions ---
        if 'Description' in new_data.columns:
            new_data['Category'] = new_data['Description'].apply(self._get_category)
        else:
            new_data['Category'] = 'Other'

        # --- 5. CURRENCY NORMALIZATION ---
        #Example: <100 treated as USD, >=100 as PKR (adjust if needed)
        conversion_rate_usd_to_pkr = 280  # 1 USD = 280 PKR, change as needed
        new_data['Currency'] = new_data['Amount'].apply(lambda x: 'USD' if x < 100 else 'PKR')

        # Convert all amounts to PKR
        new_data['Amount_PKR'] = new_data.apply(
              lambda row: row['Amount'] * conversion_rate_usd_to_pkr if row['Currency'] == 'USD' else row['Amount'],
              axis=1
         )

            # Replace original Amount column with normalized Amount
        new_data = new_data.drop(columns=['Amount'])
        new_data = new_data.rename(columns={'Amount_PKR': 'Amount'})

         # --- 6. Merge with existing data ---
        if self.data is None:
            self.data = new_data
        else:
            self.data = pd.concat([self.data, new_data], ignore_index=True)

        print(f"Loaded {len(new_data)} transactions from {filepath}.")
        return True

    def train_forecast_model(self):
        """Trains a regression model on monthly aggregates."""
        if self.data is None or self.data.empty:
            print("No data to train on.")
            return

        # Create a working copy to avoid messing up the original data
        df_train = self.data.copy()
        
        # Sort by date
        df_train = df_train.sort_values('Date')

        # Set the index to Date. This is the step that was failing before.
        # We did the conversion in load_csv, so this is now safe.
        df_train = df_train.set_index('Date')

        # 1. Aggregate data by Month
        # 'ME' = Month End.
        try:
            monthly_data = df_train.resample('ME')['Amount'].sum().reset_index()
        except TypeError:
            # Fallback for older pandas versions
            monthly_data = df_train.resample('M')['Amount'].sum().reset_index()
        
        # 2. Create 'Time' feature (0, 1, 2...) for the ML model
        monthly_data['Month_Index'] = np.arange(len(monthly_data))
        
        # 3. Train Linear Regression
        X = monthly_data[['Month_Index']]
        y = monthly_data['Amount']
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        self.monthly_aggregates = monthly_data
        print("Model trained on your historical spending.")

    def predict_future_spending(self, months_ahead=3):
        """Predicts spending for the next N months."""
        if self.model is None:
            print("Please train the model first.")
            return

        last_month_index = self.monthly_aggregates['Month_Index'].max()
        future_indices = np.arange(last_month_index + 1, last_month_index + 1 + months_ahead).reshape(-1, 1)
        
        future_df = pd.DataFrame(future_indices, columns=['Month_Index'])
        predictions = self.model.predict(future_df)
        print(f"\n---Forecast for next {months_ahead} months ---")
        for i, pred in enumerate(predictions):
            last_date = self.monthly_aggregates['Date'].max()
            month_date = last_date + pd.DateOffset(months=i+1)
            print(f"{month_date.strftime('%B %Y')}: Predicted Spend ${pred:.2f}")
        
        return predictions
    
    def analyze_spending_by_category(self):
        """Prints a breakdown of spending by category."""
        if self.data is None or self.data.empty:
            return

        print("\n---Spending Breakdown by Category ---")
        
        # Group by Category and sum the Amount
        # sort_values puts the biggest spenders at the top
        breakdown = self.data.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        
        total_spent = breakdown.sum()

        for category, amount in breakdown.items():
            percentage = (amount / total_spent) * 100
            print(f"{category}: ${amount:.2f} ({percentage:.1f}%)")

    def plot_category_pie_chart(self):
        """Generates a pie chart of your spending."""
        if self.data is None or self.data.empty: return

        breakdown = self.data.groupby('Category')['Amount'].sum()
        
        # Create the chart
        plt.figure(figsize=(8, 8))
        plt.pie(breakdown, labels=breakdown.index, autopct='%1.1f%%', startangle=140)
        plt.title('My Spending Breakdown')
        plt.show() # This opens a window with the chart

    def train_category_classifier(self):
        """Trains a Naive Bayes model using your keywords as the 'Textbook'."""
        print("Training the AI Classifier...")
        
        # 1. Prepare the Training Data
        # We need two lists: one for words (X) and one for labels (y)
        training_texts = []
        training_labels = []
        
        for category, keywords in self.category_keywords.items():
            for word in keywords:
                training_texts.append(word)
                training_labels.append(category)
                
        # 2. Build the Pipeline
        # CountVectorizer: Converts words to number counts (Bag of Words)
        # MultinomialNB: The Probability Machine
        self.classifier = make_pipeline(CountVectorizer(), MultinomialNB())
        
        # 3. Train the model
        self.classifier.fit(training_texts, training_labels)
        print("AI Brain is ready! It can now guess categories.")  


    def teach_the_bot(self):
        """Finds unknown transactions and asks the user to categorize them."""
        if self.data is None or self.data.empty: 
            return

        # Find all unique descriptions that were labeled as 'Other'
        unknowns = self.data[self.data['Category'] == 'Other']['Description'].unique()
        
        if len(unknowns) == 0:
            print("The AI recognized everything! No teaching needed today.")
            return

        print(f"\n--- Interactive Learning Mode ---")
        print(f"I found {len(unknowns)} unknown types of transactions.")
        
        learned_something_new = False

        # Loop through each unknown item and ask the user
        for desc in unknowns:
            print(f"\nUnknown Transaction: '{desc}'")
            print(f"Current Categories: {list(self.category_keywords.keys())}")
            
            # This pauses the program and waits for you to type an answer
            user_input = input("Which category does this belong to? (Type 'skip' to ignore, or type a new category): ").strip()
            
            if user_input.lower() != 'skip' and user_input != '':
                # Capitalize nicely (e.g., 'food' becomes 'Food')
                new_cat = user_input.capitalize() 
                
                # 1. Add to the dictionary
                if new_cat not in self.category_keywords:
                    self.category_keywords[new_cat] = [] # Create category if it doesn't exist
                
                self.category_keywords[new_cat].append(str(desc).lower())
                
                # 2. Update the dataframe so the pie chart is accurate
                self.data.loc[self.data['Description'] == desc, 'Category'] = new_cat
                
                learned_something_new = True
                print(f"Got it! '{desc}' is now '{new_cat}'.")

        # If we taught it new things, save to the JSON file and retrain!
        if learned_something_new:
            self.save_knowledge() # <--- NEW: Save to hard drive!
            self.train_category_classifier() 
            print("🧠 AI Brain updated and saved to hard drive! I will remember this forever.")

    def check_health_and_alerts(self):
        """Checks for anomalies and budget overflow."""
        if self.data is None or self.data.empty: return

        current_month = datetime.now().strftime('%Y-%m')
        
        # Filter for data in the current month
        # We use the copy again to be safe
        df_check = self.data.copy()
        current_month_data = df_check[df_check['Date'].dt.strftime('%Y-%m') == current_month]
        
        total_spent_now = current_month_data['Amount'].sum()
        remaining = self.budget_limit - total_spent_now
        
        print(f"\n--- Health Check ({current_month}) ---")
        print(f"Budget Limit: ${self.budget_limit}")
        print(f"Spent this month (in CSV): ${total_spent_now:.2f}")
        
        if remaining > 0:
            print(f" Status: Safe. You can spend ${remaining:.2f} more.")
        else:
            print(f" ALERT: Over budget by ${abs(remaining):.2f}!")

        # Anomaly Detection
        if hasattr(self, 'monthly_aggregates'):
            mean_spend = self.monthly_aggregates['Amount'].mean()
            std_dev = self.monthly_aggregates['Amount'].std()
            
            if std_dev > 0:
                threshold = mean_spend + (1.5 * std_dev)
                
                # Check the LAST recorded month in your CSV
                last_recorded = self.monthly_aggregates.iloc[-1]
                last_spend = last_recorded['Amount']
                last_month_name = last_recorded['Date'].strftime('%B')

                print(f"\n--- Statistical Analysis ---")
                print(f"Your Average Monthly Spend: ${mean_spend:.2f}")
                print(f"Spending in {last_month_name}: ${last_spend:.2f}")
                
                if last_spend > threshold:
                    print(f"WARNING: {last_month_name} spending was unusually high!")
    
   
   

            

# ==========================================
# REAL DATA EXECUTION
# ==========================================
if __name__ == "__main__":
   
    default_budget = 50000

    choice = input("Do you want to enter budget? (y/n): ")

    if choice.lower() == "y":
         budget = float(input("Enter your budget: "))
    else:
         budget = default_budget


    bot = FinanceAssistant(budget_limit=budget)

    bot.train_category_classifier() 

    print("Initializing System...")
    file_name = "my_expenses.csv"
    
    if bot.load_csv(file_name):
        
        # --- NEW CALL: Trigger the Interactive Teacher! ---
        bot.teach_the_bot() 
        # --------------------------------------------------
        
        bot.train_forecast_model()              # Train the spending forecast model
        bot.check_prediction_accuracy()         # 🔹 NEW: Check accuracy of predictions
        bot.analyze_spending_by_category()      # Analyze category breakdown
        bot.plot_category_pie_chart()           # Plot spending pie chart
        bot.predict_future_spending(months_ahead=3)  # Predict next 3 months
        bot.check_health_and_alerts()           # Check budget alerts
    else:
        print("System stopped. Please create the CSV file.")
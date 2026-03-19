from ml.anomaly_detector import detect_anomalies
from core.assistant import FinanceAssistant
import sys
import time

def print_header():
    """Prints a beautiful welcome header for the application."""
    print("\n" + "="*55)
    print("AI PERSONAL FINANCE TRACKER v2.0")
    print("="*55)

def main():
    print_header()

    # 1. Budget Initialization with Crash Protection
    default_budget = 50000.0
    budget = default_budget

    choice = input("Do you want to set a custom monthly budget? (y/n): ").strip().lower()
    
    if choice == 'y':
        while True:
            try:
                budget_input = input("Enter your monthly budget ($): ")
                # Clean up any accidental commas or dollar signs
                budget = float(budget_input.replace(',', '').replace('$', ''))
                if budget <= 0:
                    print("Budget must be greater than zero.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a valid number (e.g., 50000).")

    print(f"\nMonthly Budget set to: ${budget:,.2f}")
    
    # 2. Boot up the Assistant
    time.sleep(0.5) # Gives a cool, slight loading effect
    bot = FinanceAssistant(budget_limit=budget)

    # 3. Load Data and Train the Predictor
    if not bot.load_data():
        print("\nCRITICAL: Could not load expense data. Shutting down.")
        sys.exit(1)

    # Automatically train the forecaster in the background
    bot.train_forecast()
    # 4. The Interactive Main Menu Loop
    while True:
        print("\n" + "="*55)
        print("MAIN MENU")
        print("="*55)
        print("[1] View Spending Breakdown")
        print("[2] Run Health Check (Current vs Budget)")
        print("[3] AI Financial Forecast & Accuracy")
        print("[4] Show Interactive Donut Chart")
        print("[5] Teach AI a New Category Keyword")
        print("[6] Run AI Anomaly Detection")
        print("[7] Set Category-Specific Budgets")
        print("[8] Exit Program")
        print("-" * 55)
        
        menu_choice = input("Enter your choice (1-8): ").strip()

        if menu_choice == '1':
            bot.analyze_spending()
        elif menu_choice == '2':
            bot.health_check()
        elif menu_choice == '3':
            bot.check_prediction_accuracy()
            bot.predict_future(months=3)
        elif menu_choice == '4':
            bot.show_pie_chart()
        elif menu_choice == '5':
            bot.teach_the_bot()
        elif menu_choice == '6':
           detect_anomalies(bot.data)  
        elif menu_choice == '7':
            # Show existing categories to help the user choose
            if bot.data is not None:
                cats = bot.data['Category'].unique()
                print(f"\nExisting Categories: {', '.join(cats)}")
                
                target_cat = input("Enter the category to set a budget for: ").strip()
                try:
                    limit_amt = float(input(f"Enter monthly limit for {target_cat} ($): "))
                    bot.set_category_budget(target_cat, limit_amt)
                except ValueError:
                    print("Invalid amount. Please enter a number.")
            else:
                print("Please load data first.")    
        elif menu_choice == '8':
            print("\nSaving AI memory and shutting down. Have a great day!\n")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 8.")


if __name__ == "__main__":
    main()
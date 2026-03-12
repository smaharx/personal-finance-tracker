from core.assistant import FinanceAssistant

def main():

    default_budget = 50000

    choice = input("Do you want to enter budget? (y/n): ")

    if choice.lower() == "y":
        budget = float(input("Enter your budget: "))
    else:
        budget = default_budget

    bot = FinanceAssistant(budget_limit=budget)

    print("Initializing System...")

    if bot.load_data("data/my_expenses.csv"):

        bot.teach_the_bot()

        bot.train_forecast()

        bot.check_prediction_accuracy()

        bot.analyze_spending()

        # bot.show_pie_chart()

        bot.predict_future(months=3)

        bot.health_check()

    else:
        print("CSV file not found.")


if __name__ == "__main__":
    main()
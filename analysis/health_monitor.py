from datetime import datetime


def health_check(data, budget_limit, monthly_data):

    current_month = datetime.now().strftime("%Y-%m")

    month_data = data[data["Date"].dt.strftime("%Y-%m") == current_month]

    spent = month_data["Amount"].sum()

    remaining = budget_limit - spent

    print("Budget:", budget_limit)
    print("Spent:", spent)

    if remaining > 0:
        print("Safe:", remaining)
    else:
        print("Over budget:", abs(remaining))
import pandas as pd

def health_check(data, budget_limit, monthly_aggregates):
    print("\n" + "="*45)
    print("FINANCIAL HEALTH DASHBOARD")
    print("="*45)

    if data is None or data.empty:
        print("No data available for health check.")
        return

    # BUG FIX: Find the most recent month in your actual data, 
    # not the real-world calendar. This prevents the $0.0 bug!
    latest_date = data["Date"].max()
    latest_month_str = latest_date.strftime("%Y-%m")
    display_month = latest_date.strftime("%B %Y")

    # Filter data for that specific month
    month_data = data[data["Date"].dt.strftime("%Y-%m") == latest_month_str]
    spent = month_data["Amount"].sum()
    remaining = budget_limit - spent
    
    # Calculate percentage for the visual bar
    if budget_limit > 0:
        percent_spent = (spent / budget_limit) * 100
    else:
        percent_spent = 0

    # Print the beautifully formatted dashboard
    print(f"Active Month : {display_month}")
    print(f"Total Budget : ${budget_limit:,.2f}")
    print(f"Total Spent  : ${spent:,.2f}")
    print("-" * 45)

    if remaining >= 0:
        print(f"Status       : SAFE")
        print(f"Remaining    : ${remaining:,.2f}")
    else:
        print(f"Status       : OVER BUDGET!")
        print(f"Overspent by : ${abs(remaining):,.2f}")

    # Bonus: Draw an ASCII Progress Bar in the terminal
    bar_length = 20
    filled_length = int(bar_length * min(percent_spent, 100) / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # Change color-coding text based on how much is spent
    if percent_spent > 100:
        print(f"\nBudget Used: [{bar}] 🔴 {percent_spent:.1f}% (DANGER)")
    elif percent_spent > 85:
        print(f"\nBudget Used: [{bar}] 🟡 {percent_spent:.1f}% (WARNING)")
    else:
        print(f"\nBudget Used: [{bar}] 🟢 {percent_spent:.1f}% (GOOD)")
        
    print("="*45 + "\n")
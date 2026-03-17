import pandas as pd

def health_check(bot): # Changed parameter name to 'bot' for clarity
    print("\n" + "="*45)
    print("AI FINANCIAL HEALTH DASHBOARD")
    print("="*45)

    if bot.data is None or bot.data.empty:
        print("No data loaded to perform health check.")
        return

    # 1. Prepare Data
    bot.data['Date'] = pd.to_datetime(bot.data['Date'])
    current_month = bot.data['Date'].max().month
    current_year = bot.data['Date'].max().year
    
    monthly_data = bot.data[(bot.data['Date'].dt.month == current_month) & 
                             (bot.data['Date'].dt.year == current_year)]

    total_spent = monthly_data['Amount'].sum()
    
    # 2. Total Budget Check (Using 'budget_limit' to match assistant.py)
    print(f"Active Month : {bot.data['Date'].max().strftime('%B %Y')}")
    print(f"Total Budget : ${bot.budget_limit:,.2f}")
    print(f"Total Spent  : ${total_spent:,.2f}")
    
    status = "SAFE" if total_spent <= bot.budget_limit else "OVER BUDGET!"
    print(f"Status       : {status}")
    print("-" * 45)

    # 3. Granular Category Check
    if bot.category_budgets:
        print("CATEGORY SPECIFIC BREAKDOWN:")
        cat_spent = monthly_data.groupby('Category')['Amount'].sum()

        for cat, limit in bot.category_budgets.items():
            spent = cat_spent.get(cat, 0)
            diff = limit - spent
            percent = (spent / limit) * 100 if limit > 0 else 0
            
            icon = "✅" if diff >= 0 else "🔴"
            status_text = "Safe" if diff >= 0 else "OVERSPENT!"
            
            print(f"{icon} {cat:<12}: Spent ${spent:>9,.2f} / Limit ${limit:>9,.2f} ({percent:>5.1f}%) - {status_text}")
    else:
        print("(No category-specific budgets set yet.)")

    print("="*45 + "\n")
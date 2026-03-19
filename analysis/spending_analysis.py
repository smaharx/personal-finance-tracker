import pandas as pd

def spending_breakdown(df, budgets=None):
    """
    Analyzes spending and compares it against set budgets.
    Shows categories even if $0 has been spent.
    """
    if budgets is None:
        budgets = {}

    print("\n" + "="*65)
    print(" CATEGORY SPENDING & BUDGET BREAKDOWN")
    print("="*65)

    # 1. Calculate actual spending from the database
    if df is not None and not df.empty and 'Category' in df.columns:
        spent_data = df.groupby('Category')['Amount'].sum().to_dict()
    else:
        spent_data = {}

    # 2. Find ALL categories (combine what we spent AND what we budgeted)
    all_categories = set(list(spent_data.keys()) + list(budgets.keys()))

    # 3. Print the new, advanced table header
    print(f"{'Category':<16} | {'Spent':<12} | {'Budget':<12} | {'Status'}")
    print("-" * 65)

    # 4. Loop through every category and compare
    total_spent = 0
    for cat in sorted(all_categories):
        spent = spent_data.get(cat, 0.0)
        budget = budgets.get(cat, 0.0)
        total_spent += spent

        # Determine the health status
        status = ""
        if budget > 0:
            if spent > budget:
                status = " OVER BUDGET!"
            else:
                status = f" ${(budget - spent):,.2f} left"
        else:
            status = "No Budget Set"

        budget_str = f"${budget:,.2f}" if budget > 0 else "---"
        print(f"{cat:<16} | ${spent:<11,.2f} | {budget_str:<12} | {status}")

    print("-" * 65)
    print(f"{'TOTAL SPENT':<16} | ${total_spent:<11,.2f} |")
    print("="*65)
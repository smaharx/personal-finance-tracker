import pandas as pd

def spending_breakdown(data):
    """Prints a beautifully formatted breakdown of spending by category."""
    print("\n" + "="*50)
    print("CATEGORY SPENDING BREAKDOWN")
    print("="*50)

    # Safety Check 1: Is there actually data?
    if data is None or data.empty:
        print("No data available to analyze.")
        print("="*50 + "\n")
        return

    # Group data by category and sum the amounts
    breakdown = data.groupby("Category")["Amount"].sum()
    
    # Filter out any negative or zero amounts just in case
    breakdown = breakdown[breakdown > 0]
    total = breakdown.sum()
    
    # Safety Check 2: Prevent a "Division by Zero" error
    if total == 0:
        print("Total spending is $0.00. Nothing to break down!")
        print("="*50 + "\n")
        return

    # Sort from highest spending to lowest
    breakdown = breakdown.sort_values(ascending=False)

    # Print the table header
    print(f"{'Category':<18} | {'Amount':<13} | {'% of Total'}")
    print("-" * 50)

    # Print each category row nicely formatted
    for category, amount in breakdown.items():
        pct = (amount / total) * 100
        # <18 means align left with 18 spaces. >11 means align right with 11 spaces.
        print(f"{category:<18} | ${amount:>11,.2f} | {pct:>7.1f}%")

    # Print the final total
    print("-" * 50)
    print(f"{'TOTAL SPENDING':<18} | ${total:>11,.2f} |  100.0%")
    print("="*50 + "\n")
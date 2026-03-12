def spending_breakdown(data):

    breakdown = data.groupby("Category")["Amount"].sum()

    total = breakdown.sum()

    for category, amount in breakdown.items():

        pct = (amount / total) * 100

        print(category, amount, pct)
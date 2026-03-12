import matplotlib.pyplot as plt


def plot_pie(data):

    breakdown = data.groupby("Category")["Amount"].sum()

    plt.figure(figsize=(8,8))

    plt.pie(breakdown, labels=breakdown.index, autopct="%1.1f%%")

    plt.show()
import matplotlib.pyplot as plt
import numpy as np

def plot_pie(data):
    """Generates a modern, professional donut chart of expenses by category."""
    if data is None or data.empty:
        print("No data available to plot.")
        return

    # Group data by category and sum the amounts
    breakdown = data.groupby("Category")["Amount"].sum()
    
    # Filter out any zero or negative values (just to be safe)
    breakdown = breakdown[breakdown > 0]
    
    if breakdown.empty:
        print("No positive expenses to plot.")
        return

    # Sort values so the biggest expenses are always first
    breakdown = breakdown.sort_values(ascending=False)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Generate a nice, varied color palette based on how many categories you have
    colors = plt.cm.Paired(np.linspace(0, 1, len(breakdown)))
    
    # "Explode" (pop out) the largest slice slightly to highlight it
    explode = [0.05] + [0] * (len(breakdown) - 1)
    
    # Plot the chart with upgraded styling
    wedges, texts, autotexts = ax.pie(
        breakdown, 
        labels=breakdown.index, 
        autopct="%1.1f%%", 
        startangle=140, 
        colors=colors,
        explode=explode,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )

    # Draw a white circle in the center to turn the pie chart into a sleek Donut Chart
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    # Add a professional title
    plt.title("Spending Breakdown by Category", fontsize=16, fontweight='bold', pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a perfect circle
    ax.axis('equal')  
    plt.tight_layout()
    
    print("\nPopping open your spending chart... (Close the window to continue)")
    plt.show()
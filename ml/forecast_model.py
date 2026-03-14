import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_model(data):
    """Trains a Linear Regression model on monthly spending trends."""
    if data is None or data.empty:
        return None, None

    df = data.copy()
    df = df.sort_values("Date")
    df = df.set_index("Date")
    
    # Group data by month
    try:
        monthly = df.resample("ME")["Amount"].sum().reset_index()
    except TypeError:
        monthly = df.resample("M")["Amount"].sum().reset_index()
        
    # Safety Check: ML needs at least 2 data points (months) to draw a trendline!
    if len(monthly) < 2:
        print("Not enough historical data to train the forecasting model. (Need at least 2 months)")
        return None, monthly

    monthly["Month_Index"] = np.arange(len(monthly))
    X = monthly[["Month_Index"]]
    y = monthly["Amount"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, monthly


def predict_future(model, monthly_aggregates, months_ahead=3):
    """Predicts future spending based on the trained model."""
    print("\n" + "="*45)
    print(f"FINANCIAL FORECAST (Next {months_ahead} Months)")
    print("="*45)

    if model is None or monthly_aggregates is None or len(monthly_aggregates) < 2:
        print("Cannot generate forecast. Need more historical data.")
        print("="*45 + "\n")
        return

    last_month_index = monthly_aggregates['Month_Index'].max()
    future_indices = np.arange(last_month_index + 1, last_month_index + 1 + months_ahead).reshape(-1, 1)
    future_df = pd.DataFrame(future_indices, columns=['Month_Index'])
    
    predictions = model.predict(future_df)
    
    for i, pred in enumerate(predictions):
        last_date = monthly_aggregates['Date'].max()
        month_date = last_date + pd.DateOffset(months=i+1)
        # Prevent the AI from predicting negative spending
        display_pred = max(pred, 0)
        print(f"{month_date.strftime('%B %Y'):<15} : Predicted Spend ${display_pred:,.2f}")
    
    print("="*45 + "\n")
    return predictions


def check_accuracy(model, monthly_aggregates):
    """Tests how accurate the AI's predictions are on historical data."""
    print("\n" + "="*45)
    print("AI MODEL ACCURACY REPORT")
    print("="*45)

    if model is None or monthly_aggregates is None or len(monthly_aggregates) < 2:
        print("Model not trained. Need more historical data to check accuracy.")
        print("="*45 + "\n")
        return

    X = monthly_aggregates[['Month_Index']]
    y_real = monthly_aggregates['Amount']
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y_real, y_pred)
    mean_actual = y_real.mean()
    
    # Calculate percentage error safely
    if mean_actual > 0:
        pct_error = (mae / mean_actual) * 100
    else:
        pct_error = 0

    print(f"Average Margin of Error : ${mae:,.2f}")
    
    # Give a dynamic performance grade based on the error percentage
    if pct_error < 10:
        print(f"Accuracy Grade          : EXCELLENT ({pct_error:.1f}% error)")
    elif pct_error < 20:
        print(f"Accuracy Grade          : GOOD ({pct_error:.1f}% error)")
    else:
        print(f"Accuracy Grade          : NEEDS MORE DATA ({pct_error:.1f}% error)")
        
    print("="*45 + "\n")
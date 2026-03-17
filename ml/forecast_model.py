import pandas as pd
from sklearn.metrics import mean_absolute_error
import logging

# Suppress Prophet's background chatter to keep the terminal clean
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

try:
    from prophet import Prophet
except ImportError:
    print("Error: Prophet library not found. Please run: pip install prophet")
    Prophet = None

def train_model(data):
    """Trains a Meta Prophet model on monthly spending trends."""
    if Prophet is None or data is None or data.empty:
        return None, None

    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Group data by month
    try:
        monthly = df.set_index('Date').resample("ME")["Amount"].sum().reset_index()
    except TypeError:
        monthly = df.set_index('Date').resample("M")["Amount"].sum().reset_index()
        
    if len(monthly) < 2:
        print("Warning: Not enough historical data to train Prophet. (Need at least 2 months)")
        return None, monthly

    # PROPHET REQUIREMENT: Columns MUST be named 'ds' (datestamp) and 'y' (target)
    prophet_df = monthly.rename(columns={"Date": "ds", "Amount": "y"})
    
    # Initialize and train Prophet 
    # (We turn off yearly seasonality because we don't have multiple years of data yet)
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    
    return model, prophet_df

def predict_future(model, prophet_df, months_ahead=3):
    """Predicts future spending ranges using Meta Prophet."""
    print(f"\n--- PROPHET AI FORECAST (Next {months_ahead} Months) ---")

    if model is None or prophet_df is None or len(prophet_df) < 2:
        print("Error: Cannot generate forecast. Model not trained.\n")
        return

    # Ask Prophet to generate future dates ('MS' = Month Start)
    future = model.make_future_dataframe(periods=months_ahead, freq='MS')
    forecast = model.predict(future)
    
    # Filter out the past dates so we only print the future predictions
    future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]
   # 1. Print the Header
    print("\n" + "="*75)
    print("PROPHET AI FORECAST (Next 3 Months)")
    print("="*75)
    print(f"{'Month':<14} | {'Estimated Spend':<16} | {'Safe Low End':<15} | {'Danger High End':<16}")
    print("-" * 75)

    # 2. Loop through the forecast and print the aligned rows
    for _, row in future_forecast.iterrows():
        month_date = row['ds']
        # Prophet generates yhat (guess), yhat_lower (minimum), and yhat_upper (maximum)
        pred = max(row['yhat'], 0)
        lower = max(row['yhat_lower'], 0)
        upper = max(row['yhat_upper'], 0)
        
        month_str = month_date.strftime('%B %Y')
        
        # The <14 aligns text left, the >15 aligns numbers right so decimals stack!
        print(f"{month_str:<14} | ${pred:>15,.2f} | ${lower:>14,.2f} | ${upper:>15,.2f}")

    # 3. Print the Footer
    print("="*75 + "\n")
    
    return forecast

def check_accuracy(model, prophet_df):
    """Tests Prophet's accuracy on historical data."""
    print("\n--- PROPHET MODEL ACCURACY REPORT ---")

    if model is None or prophet_df is None or len(prophet_df) < 2:
        print("Error: Model not trained. Cannot check accuracy.\n")
        return

    # Prophet tests itself against your historical data
    forecast = model.predict(prophet_df)
    
    y_real = prophet_df['y']
    y_pred = forecast['yhat']
    
    mae = mean_absolute_error(y_real, y_pred)
    mean_actual = y_real.mean()
    
    if mean_actual > 0:
        pct_error = (mae / mean_actual) * 100
    else:
        pct_error = 0

    print(f"Average Margin of Error : ${mae:,.2f}")
    
    if pct_error < 10:
        print(f"Accuracy Grade          : EXCELLENT ({pct_error:.1f}% error)\n")
    elif pct_error < 20:
        print(f"Accuracy Grade          : GOOD ({pct_error:.1f}% error)\n")
    else:
        print(f"Accuracy Grade          : NEEDS MORE DATA ({pct_error:.1f}% error)\n")
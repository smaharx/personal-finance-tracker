import streamlit as st
import pandas as pd
import sqlite3
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import sqlite3
import os
import pandas as pd
import datetime

def generate_smart_insights(df):
    """Analyzes the dataframe and returns plain-English financial alerts."""
    
    # If there's no data yet, return a default message
    if df.empty:
        return []

    # Ensure Date column is in the correct datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    insights = []
    
    # --- 1. THE GREEN INSIGHT: Budget Tracking ---
    current_month = datetime.datetime.now().month
    current_year = datetime.datetime.now().year
    
    # Get all spending for the current month
    this_month_df = df[(df['Date'].dt.month == current_month) & (df['Date'].dt.year == current_year)]
    monthly_spend = this_month_df['Amount'].sum()
    
    # Let's set a dummy baseline budget of $2000 for now
    budget = 2000.00 
    if monthly_spend < budget:
        remaining = budget - monthly_spend
        insights.append({
            "type": "success", 
            "msg": f"🟢 **On Track:** You have spent \${monthly_spend:.2f} this month. You are **\${remaining:.2f} under budget**. Keep it up!"
        })
    else:
        overage = monthly_spend - budget
        insights.append({
            "type": "error", 
            "msg": f"🔴 **Over Budget:** You have spent \${monthly_spend:.2f}, exceeding your \${budget} budget by \${overage:.2f}."
        })

    # --- 2. THE RED INSIGHT: Category Warning ---
    if not this_month_df.empty:
        # Find the category they spent the most on this month
        top_category = this_month_df.groupby('Category')['Amount'].sum().idxmax()
        top_amount = this_month_df.groupby('Category')['Amount'].sum().max()
        
        # If they spent more than $300 in one category, trigger a warning
        if top_amount > 300:
            insights.append({
                "type": "error", 
                "msg": f"🔴 **Warning:** Based on your current speed, your spending on **{top_category}** is critically high (${top_amount:.2f}). Slow down!"
            })

    # --- 3. THE YELLOW INSIGHT: Weekend Behavior ---
    # Day of week: Monday=0, Sunday=6. Weekends are 5 and 6.
    weekends_df = df[df['Date'].dt.dayofweek >= 5]
    if not weekends_df.empty:
        # Calculate average weekend spending
        avg_weekend = weekends_df.groupby(weekends_df['Date'].dt.isocalendar().week)['Amount'].sum().mean()
        insights.append({
            "type": "warning", 
            "msg": f"🟡 **Pattern Alert:** You usually spend around **${avg_weekend:.2f}** on weekends. Keep an eye on your upcoming weekend plans."
        })

    return insights

# Import your AI function from your backend!
try:
    from finance_main import predict_category
except ImportError:
    st.error("Could not import predict_category. Make sure finance_main.py is in the same folder!")

# 1. Setup the Webpage
st.set_page_config(page_title="AI Finance Tracker", page_icon="💸", layout="wide")
st.title("💸 AI Personal Finance Dashboard")
st.markdown("Welcome to Phase 3. Your AI backend is officially connected to the web.")

# 2. Connect to your SQLite Database safely
@st.cache_data # This tells Streamlit to remember the data so it's super fast!
# 2. Connect to your SQLite Database safely
@st.cache_data
def load_data():
    conn = sqlite3.connect('data/expenses.db')
    # NEW: We pull 'rowid as ID' so we can uniquely target rows for correction
    df = pd.read_sql_query("SELECT rowid as ID, * FROM transactions", conn)
    conn.close()
    return df

def init_db():
    """Creates the database and table if they don't exist yet."""
    # Ensure the 'data' folder exists
    os.makedirs('data', exist_ok=True)
    
    # Connect to (or create) the database
    conn = sqlite3.connect('data/expenses.db')
    cursor = conn.cursor()
    
    # Create the table if it's a brand new installation
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            Date TEXT,
            Description TEXT,
            Category TEXT,
            Amount REAL
        )
    ''')
    conn.commit()
    conn.close()

# RUN THIS ONCE EVERY TIME THE APP STARTS:
init_db()


@st.cache_data
def generate_forecast(df, days=30):
    # 1. Prepare data for Prophet (Requires 'ds' for dates, 'y' for values)
    daily_spend = df.groupby('Date')['Amount'].sum().reset_index()
    daily_spend.columns = ['ds', 'y']
    
    # 2. Initialize and train the AI
    m = Prophet(daily_seasonality=True, yearly_seasonality=False)
    m.fit(daily_spend)
    
    # 3. Predict the future
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    
    return daily_spend, forecast    

# 3. Fetch and Display the Data
try:
    df = load_data()
    
    # --- NEW: CREATE UI TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 AI Forecasting", "🧠 Teach AI"])
    
    # --- TAB 1: The Main View ---
    with tab1:
        # Create two columns layout inside the tab
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Recent Transactions")
            st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True)
            
        with col2:
            st.subheader("Quick Stats")
            st.metric(label="Total Transactions", value=len(df))
            st.metric(label="Total Spent", value=f"${df['Amount'].sum():,.2f}")

    # --- TAB 2: Prophet Predictions ---
    
    with tab2:
        st.subheader("🔮 Future Expense Forecasting")
        st.markdown("Use Facebook Prophet AI to predict your spending trends for the next 30 days.")
        
        # --- NEW SMART INSIGHTS ENGINE UI ---
        st.markdown("---")
        st.subheader("💡 AI Smart Insights")
        
        insights = generate_smart_insights(df)
        
        if not insights:
            st.info("Add some expenses to see your AI financial insights!")
        else:
            for insight in insights:
                if insight["type"] == "success":
                    st.success(insight["msg"])
                elif insight["type"] == "error":
                    st.error(insight["msg"])
                elif insight["type"] == "warning":
                    st.warning(insight["msg"])
                    
        st.markdown("---")
        # --- END SMART INSIGHTS ENGINE UI ---

        if st.button("Run AI Forecast", type="primary"):
            with st.spinner("AI is analyzing your spending patterns..."):
                try:
                    # Run the ML model
                    actual_data, forecast_data = generate_forecast(df, days=30)
                    
                    # Build the interactive chart
                    fig = go.Figure()
                    
                    # Plot Actual Spending (Blue Line)
                    fig.add_trace(go.Scatter(
                        x=actual_data['ds'], y=actual_data['y'], 
                        mode='lines+markers', name='Actual Spend',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Plot AI Forecast (Red Dotted Line)
                    fig.add_trace(go.Scatter(
                        x=forecast_data['ds'], y=forecast_data['yhat'], 
                        mode='lines', name='AI Prediction',
                        line=dict(color='#d62728', width=2, dash='dot')
                    ))
                    
                    # Format the chart beautifully
                    fig.update_layout(
                        title="Actual vs. Predicted Spending",
                        xaxis_title="Date",
                        yaxis_title="Amount ($)",
                        hovermode="x unified",
                        template="plotly_dark" # Matches your dark mode UI!
                    )
                    
                    # Display the chart on the web page
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Not enough data to run Prophet. Please add a few more transactions! (Error: {e})")
    # --- TAB 3: Retraining the Categorizer ---
    # --- TAB 3: Retraining the Categorizer ---
    with tab3:
        st.subheader("🧠 Teach AI (Human-in-the-Loop)")
        st.markdown("If the AI miscategorized an expense, correct it here. This creates a clean dataset for future model retraining.")
        
        # 1. Grab the 50 most recent transactions
        recent_df = df.sort_values(by="Date", ascending=False).head(50)
        
        # 2. Create a clean dropdown menu list
        txn_list = recent_df.apply(
            lambda x: f"ID: {x['ID']} | {x['Date']} | {x['Description']} - ${x['Amount']:.2f} ➔ {x['Category']}", 
            axis=1
        ).tolist()
        
        selected_txn = st.selectbox("Select a recent transaction to correct:", ["-- Choose a transaction --"] + txn_list)
        
        # 3. The Correction Form
        if selected_txn != "-- Choose a transaction --":
            # Extract the ID mathematically from the dropdown string
            txn_id = int(selected_txn.split("|")[0].replace("ID:", "").strip())
            current_cat = selected_txn.split("➔")[1].strip()
            
            with st.form("teach_ai_form"):
                st.write(f"**Current AI Guess:** `{current_cat}`")
                new_cat = st.text_input("Enter Correct Category (e.g., Food, Utilities, Transport):")
                
                if st.form_submit_button("Update Database & Log Correction", type="primary"):
                    if new_cat and new_cat.strip().lower() != current_cat.lower():
                        try:
                            # Connect to DB and update the exact row
                            conn = sqlite3.connect('data/expenses.db')
                            cursor = conn.cursor()
                            cursor.execute("UPDATE transactions SET Category = ? WHERE rowid = ?", (new_cat.strip(), txn_id))
                            conn.commit()
                            conn.close()
                            
                            # Clear UI cache and refresh instantly
                            st.cache_data.clear()
                            st.success(f"✅ Successfully updated ID {txn_id} to '{new_cat}'!")
                            time.sleep(1) # Brief pause so you can read the success message
                            st.rerun()
                        except Exception as e:
                            st.error(f"Database error: {e}")
                    else:
                        st.warning("Please enter a new, valid category.")

except Exception as e:
    st.error(f"Error loading database: {e}")

# --- 4. ADD EXPENSE WEB FORM (SIDEBAR) ---
st.sidebar.header("⚡ Add New Expense")

with st.sidebar.form("expense_form", clear_on_submit=True):
    desc = st.text_input("Description (e.g., 'Petrol', 'Chai')")
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
    submit_button = st.form_submit_button("AI Auto-Categorize & Save")

    if submit_button and desc and amount > 0:
        start_time = time.time()
        txn_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Call your AI
            predicted_cat = predict_category(desc)
            
            # Save to Database
            conn = sqlite3.connect('data/expenses.db')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO transactions (Date, Description, Amount, Category) VALUES (?, ?, ?, ?)", 
                (txn_date, desc, amount, predicted_cat)
            )
            conn.commit()
            conn.close()
            
            # Stop Stopwatch
            api_speed = time.time() - start_time
            
            # Success Message
            st.sidebar.success(f"✅ Saved as **{predicted_cat}** in {api_speed:.2f}s!")
            
            # Clear the cache so the main table updates instantly
            st.cache_data.clear()
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Error connecting to AI: {e}")
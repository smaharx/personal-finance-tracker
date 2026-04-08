import streamlit as st
import pandas as pd
import sqlite3
import time
import os
import datetime
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import plotly.express as px


# --- 1. SMART INSIGHTS ENGINE ---
def generate_smart_insights(df):
    """Analyzes the dataframe and returns plain-English financial alerts."""
    if df.empty:
        return []

    df['Date'] = pd.to_datetime(df['Date'])
    insights = []
    
    # Budget Tracking
    current_month = datetime.datetime.now().month
    current_year = datetime.datetime.now().year
    this_month_df = df[(df['Date'].dt.month == current_month) & (df['Date'].dt.year == current_year)]
    monthly_spend = this_month_df['Amount'].sum()
    
    budget = 2000.00 
    if monthly_spend < budget:
        remaining = budget - monthly_spend
        insights.append({
            "type": "success", 
            "msg": rf"🟢 **On Track:** Spent \${monthly_spend:.2f}. You are **\${remaining:.2f} under budget**. Keep it up!"
        })
    else:
        overage = monthly_spend - budget
        insights.append({
            "type": "error", 
            "msg": rf"🔴 **Over Budget:** Spent \${monthly_spend:.2f}, exceeding your \${budget} budget by \${overage:.2f}."
        })

    # Category Warning
    if not this_month_df.empty:
        cat_sums = this_month_df.groupby('Category')['Amount'].sum()
        top_category = cat_sums.idxmax()
        top_amount = cat_sums.max()
        
        if top_amount > 300:
            insights.append({
                "type": "error", 
                "msg": rf"🔴 **Warning:** Spending on **{top_category}** is high (\${top_amount:.2f})."
            })

    # Weekend Behavior
    weekends_df = df[df['Date'].dt.dayofweek >= 5]
    if not weekends_df.empty:
        avg_weekend = weekends_df.groupby(weekends_df['Date'].dt.isocalendar().week)['Amount'].sum().mean()
        insights.append({
            "type": "warning", 
            "msg": rf"🟡 **Pattern Alert:** You usually spend around **\${avg_weekend:.2f}** on weekends."
        })

    return insights

# --- 2. DATABASE & AI BACKEND ---
try:
    from finance_main import predict_category
except ImportError:
    st.error("Could not import predict_category. Make sure finance_main.py is in the same folder!")

def init_db():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/expenses.db')
    cursor = conn.cursor()
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

init_db()

@st.cache_data
def load_data():
    conn = sqlite3.connect('data/expenses.db')
    df = pd.read_sql_query("SELECT rowid as ID, * FROM transactions", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def generate_forecast(df, days=30):
    daily_spend = df.groupby('Date')['Amount'].sum().reset_index()
    daily_spend.columns = ['ds', 'y']
    m = Prophet(daily_seasonality=True, yearly_seasonality=False)
    m.fit(daily_spend)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return daily_spend, forecast    
def detect_anomalies(df):
    # We only care about the amount spent for anomaly detection
    # We drop any empty values just in case
    data = df[['Amount']].dropna()
    
    # Initialize the model. 
    # 'contamination=0.05' means we assume roughly 5% of expenses are unusual/anomalies
    model = IsolationForest(contamination=0.05, random_state=42)
    
    # Train the model and get predictions (-1 means anomaly, 1 means normal)
    df['Anomaly'] = model.fit_predict(data)
    
    # Filter the dataframe to only show the anomalies
    anomalies_df = df[df['Anomaly'] == -1]
    
    return anomalies_df    

# --- 3. WEB INTERFACE SETUP ---
st.set_page_config(page_title="AI Finance Tracker", page_icon="💸", layout="wide")
st.title("💸 AI Personal Finance Dashboard")

try:
    df = load_data()
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Forecast", "Teach AI", "🚨 Alerts"])
    
    # --- TAB 1: MAIN VIEW ---
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Recent Transactions")
            st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True)
        with col2:
            st.subheader("Quick Stats")
            st.metric(label="Total Transactions", value=len(df))
            st.metric(label="Total Spent", value=rf"${df['Amount'].sum():,.2f}")

    # --- TAB 2: FORECASTING ---
    with tab2:
        st.subheader("🔮 Expense Forecasting & Insights")
        insights = generate_smart_insights(df)
        for insight in insights:
            if insight["type"] == "success": st.success(insight["msg"])
            elif insight["type"] == "error": st.error(insight["msg"])
            elif insight["type"] == "warning": st.warning(insight["msg"])
                    
        if st.button("Run AI Forecast", type="primary"):
            with st.spinner("Analyzing patterns..."):
                try:
                    actual_data, forecast_data = generate_forecast(df)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=actual_data['ds'], y=actual_data['y'], mode='lines+markers', name='Actual'))
                    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], mode='lines', name='Predicted', line=dict(dash='dot')))
                    fig.update_layout(title="Spending Forecast", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error("Add more data to unlock forecasting!")

    # --- TAB 3: TEACH AI ---
    with tab3:
        st.header("🧠 Teach the AI")
        recent_df = df.sort_values(by="Date", ascending=False).head(50)
        if not recent_df.empty:
            txn_list = recent_df.apply(lambda x: f"ID: {x['ID']} | {x['Date'].strftime('%Y-%m-%d')} | {x['Description']} - ${x['Amount']:.2f} ➔ {x['Category']}", axis=1).tolist()
            selected_txn = st.selectbox("Select a transaction to correct:", ["-- Choose --"] + txn_list)
            
            if selected_txn != "-- Choose --":
                txn_id = int(selected_txn.split("|")[0].replace("ID:", "").strip())
                current_cat = selected_txn.split("➔")[1].strip()
                st.info(f"Current AI Guess: `{current_cat}`")
                categories = ["Food", "Transport", "Shopping", "Utilities", "Entertainment", "Health", "Other"]
                new_cat = st.selectbox("Correct Category:", categories)
                
                if st.button("Save Correction"):
                    conn = sqlite3.connect('data/expenses.db')
                    cursor = conn.cursor()
                    cursor.execute("UPDATE transactions SET Category = ? WHERE rowid = ?", (new_cat, txn_id))
                    conn.commit()
                    conn.close()
                    st.cache_data.clear()
                    st.success("AI Learned! Correction Saved. 🚀")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
        else:
            st.warning("No data found.")
    with tab4:
        st.subheader("🚨 Anomaly Detection (Isolation Forest)")
        st.write("Our AI analyzes your spending patterns to detect unusual transactions.")

        # Run the algorithm on your cached data
        anomalies = detect_anomalies(df)

    
        # Create a scatter plot of ALL expenses
        fig_anomalies = px.scatter(
            df, 
            x="Date", 
            y="Amount", 
            color=df['Anomaly'].astype(str), 
            color_discrete_map={'-1': 'red', '1': 'blue'}, 
            hover_data=['Description', 'Category']
        )

        # Clean up the legend
        # Add professional titles, axis labels, and clean up the legend
        fig_anomalies.update_layout(
             title={
                'text': "Transaction History: Anomalies vs. Normal Spending",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
            xaxis_title="Timeline",
            yaxis_title="Transaction Amount",
            showlegend=False
        )

        st.plotly_chart(fig_anomalies, use_container_width=True)

        #Show a warning if anomalies exist
        if not anomalies.empty:
            st.error(f"⚠️ We detected {len(anomalies)} unusual transactions!")
            st.dataframe(anomalies[['Date', 'Description', 'Category', 'Amount']])
        else:
            st.success("✅ Your spending looks normal. No anomalies detected.")             
            

    # --- SIDEBAR: ADD EXPENSE ---
    st.sidebar.header("⚡ Add New Expense")
    with st.sidebar.form("expense_form", clear_on_submit=True):
        desc = st.text_input("Description")
        amount = st.number_input("Amount", min_value=0.0)
        submit_button = st.form_submit_button("AI Categorize & Save")

        if submit_button and desc and amount > 0:
            start_time = time.time()
            txn_date = datetime.datetime.now().strftime("%Y-%m-%d")
            try:
                # ALL LINES BELOW INDENTED CORRECTLY
                predicted_cat = predict_category(desc)
                conn = sqlite3.connect('data/expenses.db')
                cursor = conn.cursor()
                cursor.execute("INSERT INTO transactions (Date, Description, Amount, Category) VALUES (?, ?, ?, ?)", (txn_date, desc, amount, predicted_cat))
                conn.commit()
                conn.close()
                st.cache_data.clear()
                st.sidebar.success(rf"✅ Saved as {predicted_cat} in {time.time()-start_time:.2f}s!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {e}")       

except Exception as e:
    st.error(f"App Load Error: {e}")
    


      

    
    




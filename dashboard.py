import streamlit as st
import pandas as pd
import sqlite3
import time
from datetime import datetime

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
def load_data():
    conn = sqlite3.connect('data/expenses.db')
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    return df

# 3. Fetch and Display the Data
try:
    df = load_data()
    
    # Create two columns layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Recent Transactions")
        st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True)
        
    with col2:
        st.subheader("Quick Stats")
        st.metric(label="Total Transactions", value=len(df))
        st.metric(label="Total Spent", value=f"${df['Amount'].sum():,.2f}")

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
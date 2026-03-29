import streamlit as st
import pandas as pd
import sqlite3

# 1. Setup the Webpage
st.set_page_config(page_title="AI Finance Tracker", page_icon="💸", layout="wide")
st.title("💸 AI Personal Finance Dashboard")
st.markdown("Welcome to Phase 3. Your AI backend is officially connected to the web.")

# 2. Connect to your SQLite Database safely
@st.cache_data # This tells Streamlit to remember the data so it's super fast!
def load_data():
    conn = sqlite3.connect('data/expenses.db')
    # Load everything from the transactions table into a Pandas DataFrame
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
        # Display the data as a beautiful, interactive web table
        st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True)
        
    with col2:
        st.subheader("Quick Stats")
        st.metric(label="Total Transactions", value=len(df))
        st.metric(label="Total Spent", value=f"${df['Amount'].sum():,.2f}")

except Exception as e:
    st.error(f"Error loading database: {e}")
    st.info("Make sure you are running this from the root project folder!")
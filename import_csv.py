import pandas as pd
import sqlite3
import os

def import_csv_to_db(csv_filename):
    print(f"\n--- Starting Bulk Import for {csv_filename} ---")
    
    # 1. Look for the file in the data folder
    csv_path = f"data/{csv_filename}"
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find '{csv_path}'. Make sure it is inside the 'data' folder.")
        return

    # 2. Read the CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"📄 Found {len(df)} rows in your CSV.")
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    # 3. Rearrange the columns to match the SQL table structure perfectly
    try:
        df = df[['Date', 'Description', 'Amount', 'Category']]
    except KeyError as e:
        print(f"❌ Error: Missing column {e}. Check your CSV headers again.")
        return

    # 4. Save to the Real SQL Database
    print("🔌 Connecting to the SQL database...")
    conn = sqlite3.connect('data/expenses.db')
    
    # 'append' means it will add these 86 rows without deleting your Sambosa!
    df.to_sql('transactions', conn, if_exists='append', index=False)
    conn.close()
    
    print("✅ SUCCESS! All transactions securely moved to expenses.db.")

if __name__ == "__main__":
    # Pointing directly to your file
    import_csv_to_db('my_expenses.csv')
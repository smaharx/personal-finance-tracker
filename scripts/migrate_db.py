import sqlite3
import pandas as pd
import os

def migrate_csv_to_sql():
    print("Starting Database Migration...")
    
    csv_path = 'data/my_expenses.csv'
    db_path = 'data/expenses.db'

    # 1. Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: Cannot find {csv_path}")
        return

    # 2. Read the old CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} transactions in CSV.")

    # 3. Connect to the new SQLite database (this creates the file if it doesn't exist)
    conn = sqlite3.connect(db_path)

    # 4. Magically send the Pandas DataFrame into a real SQL table named 'transactions'
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    
    # 5. Verify the migration
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions")
    row_count = cursor.fetchone()[0]

    conn.close()
    
    print(f"Migration Complete! Successfully saved {row_count} rows to {db_path}")
    print("You can now safely ignore or delete your old .csv file!")

if __name__ == "__main__":
    migrate_csv_to_sql()
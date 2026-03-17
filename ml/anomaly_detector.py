import pandas as pd

def detect_anomalies(data, threshold_multiplier=3.0, min_amount=50.0):

    print("AI Anomoly Detection(Fraud & Spike Scan)---")

    if data is None or data.empty:
        print("Error: no memory available")
        return

    df = data.copy()
    anomoly_found = 0

    category_stats = df.groupby("Category")["Amount"].mean().to_dict()

    for index, row in df.iterrows():
        cat = row['Category'] 
        amount = row['Amount']

        try:
            date = pd.to_datetime(row['Date']).strftime('%b, %d, %Y')
        except Exception:
            date = str(row['Date'])   

            avg_amount =     category_stats.get(cat,0) 

            if avg_amount > 0 and amount >= (avg_amount * threshold_multiplier) and amount >=min_amount:
                print(f"ALERT: Unusually high '{cat}' expense detected.")
                print(f"   Date   : {date}")
                print(f"   Amount : ${amount:,.2f} (Normal Average: ~${avg_amount:,.2f})")
                print("-" * 45)
                anomoly_found += 1


    if anomoly_found ==0:
        print("Alert clear. No Suspisious or highly unusual spending detected.")
    else:
        print(f"Scan complete. Found {anomoly_found} anomalies.")   


    print("-" * 49 + "\n")            

          


            




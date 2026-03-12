import pandas as pd
import os

def load_csv(filepath, classifier):

    if not os.path.exists(filepath):
        print("CSV file not found.")
        return None

    data = pd.read_csv(filepath)

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    data = data.dropna(subset=["Date"])

    if "Description" in data.columns and classifier:

        data["Category"] = data["Description"].apply(
            lambda x: classifier.predict([str(x)])[0]
        )
    else:
        data["Category"] = "Other"

    return data
import numpy as np
from sklearn.linear_model import LinearRegression


def train_model(data):

    df = data.copy()

    df = df.sort_values("Date")

    df = df.set_index("Date")

    monthly = df.resample("M")["Amount"].sum().reset_index()

    monthly["Month_Index"] = np.arange(len(monthly))

    X = monthly[["Month_Index"]]
    y = monthly["Amount"]

    model = LinearRegression()

    model.fit(X, y)

    return model, monthly
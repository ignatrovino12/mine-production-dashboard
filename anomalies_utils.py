import numpy as np
import pandas as pd
from scipy.stats import zscore
from outliers import smirnov_grubbs as grubbs

def detect_iqr(data, iqr_multiplier, mine_id, mine_name):
    data = data.copy()
    values = data["Final Production"]

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr_value = q3 - q1
    lower = q1 - iqr_multiplier * iqr_value
    upper = q3 + iqr_multiplier * iqr_value
    anomalies = data[(values < lower) | (values > upper)].copy()

    if anomalies.empty:
        return anomalies

    anomalies["Mine ID"] = mine_id
    anomalies["Mine Name"] = mine_name
    anomalies["Anomaly Type"] = np.where(anomalies["Final Production"] < lower, "Low", "High")
    anomalies["Lower Bound"] = lower
    anomalies["Upper Bound"] = upper

    return anomalies

def detect_zscore(data, z_threshold, mine_id, mine_name):
    data = data.copy()
    values = data["Final Production"]

    scores = pd.Series( zscore(values, nan_policy="omit"), index=data.index)
    anomalies = data[abs(scores) > z_threshold].copy()

    if anomalies.empty:
        return anomalies

    anomalies["Mine ID"] = mine_id
    anomalies["Mine Name"] = mine_name
    anomalies["Z-score"] = scores.loc[anomalies.index]
    anomalies["Anomaly Type"] = np.where( anomalies["Z-score"] < 0, "Low", "High" )

    return anomalies

def detect_moving_average(data, ma_window, ma_threshold, mine_id, mine_name):
    data = data.copy().sort_values("Date")

    data["Moving Average"] = (
        data.set_index("Date")["Final Production"]
        .rolling(f"{ma_window}D", closed="left", min_periods=1)
        .mean()
        .to_numpy()
    )

    data["Distance (%)"] = (
        (data["Final Production"] - data["Moving Average"])
        .div(data["Moving Average"].abs().where(data["Moving Average"].ne(0)))
        .mul(100)
    )

    anomalies = data[data["Distance (%)"].abs() > ma_threshold].copy()

    if anomalies.empty:
        return anomalies

    anomalies["Mine ID"] = mine_id
    anomalies["Mine Name"] = mine_name
    anomalies["Anomaly Type"] = np.where( anomalies["Distance (%)"] < 0, "Low", "High")

    return anomalies

def detect_grubbs(data, alpha, mine_id, mine_name):
    data = data.copy()
    values = data["Final Production"].to_numpy()

    outlier_indices = (
        grubbs.two_sided_test_indices(values, alpha=alpha)
        if len(values) >= 3 and np.std(values) > 0
        else []
    )

    anomalies = data.iloc[outlier_indices].copy()

    if anomalies.empty:
        return anomalies

    anomalies["Mine ID"] = mine_id
    anomalies["Mine Name"] = mine_name
    mean_value = data["Final Production"].mean()
    anomalies["Anomaly Type"] = np.where(anomalies["Final Production"] < mean_value, "Low", "High")

    return anomalies
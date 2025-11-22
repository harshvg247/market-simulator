import pandas as pd
from statsmodels.tsa.stattools import kpss

def check_stationarity_kpss(data, significance_level=0.05):
    """
    Performs the KPSS test (Kwiatkowski–Phillips–Schmidt–Shin).

    H0: The series is STATIONARY.
    H1: The series is NON-STATIONARY.

    Args:
        data (pd.Series)
        significance_level (float)

    Returns:
        dict
    """
    print("\n--- Running KPSS Test ---")

    data_clean = data.dropna()
    
    # regression='c' = stationarity around constant
    statistic, p_value, lags, critical_values = kpss(data_clean, regression='c', nlags='auto')

    print(f"KPSS Statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Number of lags: {lags}")

    print("\nCritical Values:")
    for key, value in critical_values.items():
        print(f"   {key}: {value:.4f}")

    print("\n--- Inference ---")
    if p_value < significance_level:
        print(f"✖ We REJECT the null hypothesis (p < {significance_level}).")
        print("✖ The series likely contains a UNIT ROOT (NON-STATIONARY).")
        decision = False
    else:
        print(f"✔ We FAIL to reject the null hypothesis (p ≥ {significance_level}).")
        print("✔ No evidence against STATIONARITY (but not proof of stationarity).")
        decision = True

    return {
        "kpss_statistic": statistic,
        "p_value": p_value,
        "lags_used": lags,
        "critical_values": critical_values,
        "is_stationary": decision
    }

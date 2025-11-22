import pandas as pd
from statsmodels.tsa.stattools import adfuller

def check_stationarity_adf(data, significance_level=0.05):
    """
    Performs the Augmented Dickey-Fuller (ADF) test.
    
    H0: The series has a unit root (NON-STATIONARY)
    H1: The series is stationary.

    Args:
        data (pd.Series): Time series.
        significance_level (float): Significance threshold.

    Returns:
        dict: Results including stationarity decision.
    """
    print("\n--- Running Augmented Dickey-Fuller (ADF) Test ---")

    data_clean = data.dropna()
    adf_result = adfuller(data_clean, autolag='AIC')

    adf_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    print(f"ADF Statistic: {adf_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    print("\nCritical Values:")
    for key, value in critical_values.items():
        print(f"   {key}: {value:.4f}")

    print("\n--- Inference ---")
    if p_value < significance_level:
        print(f"We REJECT the null hypothesis (p < {significance_level}).")
        print("The series shows significant evidence of STATIONARITY.")
        decision = True
    else:
        print(f"We FAIL to reject the null hypothesis (p ≥ {significance_level}).")
        decision = False

    return {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "critical_values": critical_values,
        "is_stationary": decision
    }

import pandas as pd
from statsmodels.tsa.stattools import adfuller

def check_stationarity_adf(data, significance_level=0.05):
    """
    Performs the Augmented Dickey-Fuller (ADF) test to check
    if a time series is stationary (mean-reverting).
    
    Args:
        data (pd.Series): The time series data to test.
        significance_level (float): The p-value threshold.
        
    Returns:
        bool: True if the data is stationary, False otherwise.
    """
    print("--- Running Augmented Dickey-Fuller (ADF) Test ---")
    
    # The adfuller() function needs clean data
    data_clean = data.dropna()
    
    # Run the test
    # autolag='AIC' lets the test find the best lag
    result = adfuller(data_clean, autolag='AIC')
    
    # Get the important results
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # Print a clear report
    print(f"ADF Statistic: {adf_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    print("\nCritical Values:")
    for key, value in critical_values.items():
        print(f"   {key}: {value:.4f}")
        
    # Give a final conclusion
    if p_value < significance_level:
        print(f"\nConclusion: The data is STATIONARY (p < {significance_level}).")
        print("It is suitable for the Ornstein-Uhlenbeck model.")
        return True
    else:
        print(f"\nConclusion: The data is NON-STATIONARY (p >= {significance_level}).")
        print("It is NOT suitable for the OU model (it acts like a random walk).")
        return False
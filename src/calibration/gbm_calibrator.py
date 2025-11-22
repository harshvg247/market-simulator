import numpy as np
import pandas as pd
from .base_calibrator import Calibrator
from scipy.stats import norm # We need this for log-likelihood

class GbmCalibrator(Calibrator):
    """
    Calibrates the parameters for a Geometric Brownian Motion (GBM) model.
    
    Finds:
    mu (float): Annualized drift.
    sigma (float): Annualized volatility.
    S0 (float): The first price in the data.
    """
    
    def __init__(self, dt=1/252):
        """
        Args:
            dt (float): The time step in years (default 1/252 for daily).
        """
        super().__init__() # Initialize the base class
        self.dt = dt
        # Store the annualization factor (e.g., 252)
        self.annual_factor = 1 / dt
        
    def fit(self, data):
        """
        Calculates 'mu' and 'sigma' from the log-returns of the data.
        
        Args:
            data (pd.Series): Time series of asset prices.
            
        Returns:
            dict: A dictionary {'mu', 'sigma', 'S0'}.
        """
        if not isinstance(data, pd.Series) or data.empty:
            raise ValueError("Data must be a non-empty pandas Series.")
        
        # S0 is just the first price
        S0 = data.iloc[0]
        
        # 1. Calculate log-returns: log(S_t / S_{t-1})
        log_returns = np.log(data / data.shift(1))
        
        # Drop the first value (it will be NaN)
        log_returns = log_returns.dropna()
        
        # 2. Calculate sigma (volatility)
        # Daily volatility
        sigma_daily = log_returns.std()
        # Annualize it (vol scales with sqrt of time)
        sigma_annual = sigma_daily * np.sqrt(self.annual_factor)
        
        # 3. Calculate mu (drift)
        # This is the tricky part. We need to use the
        # full log-normal formula.
        # E[log_ret] = (mu - 0.5*sigma^2)*dt
        # So, mu = E[log_ret]/dt + 0.5*sigma^2
        mean_daily_log_return = log_returns.mean()
        mu_annual = (mean_daily_log_return * self.annual_factor) + 0.5 * sigma_annual**2
        
        # 4. Store and return parameters
        self.params = {
            'mu': mu_annual,
            'sigma': sigma_annual,
            'S0': S0
        }
        
        # --- Now, calculate stats for AIC/BIC ---
        # We need the total log-likelihood of our data *given*
        # the parameters we just found.
        
        # 1. Find the mean and std of the log-returns
        #    *predicted* by our model
        mean_log_ret = (self.params['mu'] - 0.5 * self.params['sigma']**2) * self.dt
        std_log_ret = self.params['sigma'] * np.sqrt(self.dt)
        
        # 2. Use scipy's norm.logpdf (log probability density func)
        #    to find the log-prob of *each* return
        log_likelihoods = norm.logpdf(
            log_returns, 
            loc=mean_log_ret, 
            scale=std_log_ret
        )
        
        # 3. The total log-likelihood is the sum
        self.log_likelihood_ = np.sum(log_likelihoods)
        
        # 4. Store the other metrics for the base class
        self.n_params = 2  # We found 2 params: mu and sigma
        self.n_obs = len(log_returns) # n = number of log_returns
        
        return self.params
import numpy as np
import pandas as pd
from .base_calibrator import Calibrator
from scipy.stats import linregress, norm

class OuCalibrator(Calibrator):
    """
    Calibrates the parameters for an Ornstein-Uhlenbeck (OU)
    mean-reverting process using OLS regression on the
    discretized AR(1) form.
    
    Finds 3 parameters:
    theta (float): Speed of reversion
    mu (float): Long-term mean
    sigma (float): Volatility
    X0 (float): The first price in the data
    """
    
    def __init__(self, dt=1/252):
        """
        Args:
            dt (float): The time step in years (default 1/252 for daily).
        """
        super().__init__()
        self.dt = dt
        
    def fit(self, data):
        """
        Calculates 'theta', 'mu', and 'sigma' from the data
        using linear regression.
        
        Args:
            data (pd.Series): Time series of a mean-reverting asset.
            
        Returns:
            dict: A dictionary {'theta', 'mu', 'sigma', 'X0'}.
        """
        if not isinstance(data, pd.Series) or data.empty:
            raise ValueError("Data must be a non-empty pandas Series.")
        
        # 1. Prepare the X and Y for regression
        # Y = X_t
        Y = data.values[1:]
        # X = X_{t-1}
        X = data.values[:-1]
        
        # 2. Run the OLS regression
        # This finds a, b, and r_value
        slope, intercept, r_value, p_value, std_err = linregress(X, Y)
        
        # 3. Solve for our parameters
        # b = 1 - theta*dt  =>  theta = (1 - b) / dt
        theta = (1 - slope) / self.dt
        
        # a = theta*mu*dt  =>  mu = a / (theta*dt)
        # (Handle the case where theta is zero)
        if theta == 0:
            mu = np.mean(data)
        else:
            mu = intercept / (theta * self.dt)
            
        # 4. Find sigma
        # The residuals (errors) of the regression are our
        # "sigma*sqrt(dt)*Z" term
        residuals = Y - (intercept + slope * X)
        
        # std(residuals) = sigma * sqrt(dt)
        # so, sigma = std(residuals) / sqrt(dt)
        sigma = np.std(residuals) / np.sqrt(self.dt)
        
        # 5. Store and return parameters
        self.params = {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'X0': data.iloc[0] # Starting value
        }
        
        # --- Now, calculate stats for AIC/BIC ---
        # We need the total log-likelihood
        
        # The likelihoods are just the prob. of seeing
        # those residuals, given they are N(0, sigma*sqrt(dt))
        log_likelihoods = norm.logpdf(
            residuals, 
            loc=0, 
            scale=sigma * np.sqrt(self.dt)
        )
        
        self.log_likelihood_ = np.sum(log_likelihoods)
        self.n_params = 3  # theta, mu, sigma
        self.n_obs = len(Y)
        
        return self.params
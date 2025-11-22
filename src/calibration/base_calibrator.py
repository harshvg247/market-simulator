import pandas as pd
from abc import ABC, abstractmethod
import numpy as np

# ABC means Abstract Base Class. It's a "template."
class Calibrator(ABC):
    """
    Abstract Base Class for all model calibrators.
    
    This is our "blueprint." It forces every calibrator we make
    (for GBM, OU, etc.) to have the same structure.
    """
    
    def __init__(self):
        """
        Initialize the calibrator.
        We set the results to None until .fit() is called.
        """
        self.params = None
        
        # --- Metrics for Model Comparison (AIC/BIC) ---
        # We'll store these after fitting
        self.log_likelihood_ = None # The fit "score"
        self.n_params = None      # How many params did we find? (k)
        self.n_obs = None         # How many data points did we use? (n)
    
    @abstractmethod
    def fit(self, data):
        """
        Calibrates the model parameters to the provided data.
        
        This is the main function we *must* implement for
        each new calibrator.
        
        Args:
            data (pd.Series): Time series of asset prices.
        """
        # (Child class must implement this)
        pass
        
    def get_params(self):
        """
        Returns the calibrated parameters.
        
        Returns:
            dict: The stored parameter dictionary, or None if not fitted.
        """
        if self.params is None:
            print("Warning: Model not fitted yet. Call .fit() first.")
        return self.params
        
    # --- Methods for Model Comparison ---
    
    def get_log_likelihood(self):
        """Helper to get the Log-Likelihood score."""
        if self.log_likelihood_ is None:
            raise ValueError("Must run .fit() first.")
        return self.log_likelihood_

    def get_aic(self):
        """
        Returns the Akaike Information Criterion (AIC).
        Used to compare models. Lower is better.
        
        AIC = 2*k - 2*L
        k = number of parameters
        L = log-likelihood
        """
        if self.log_likelihood_ is None:
            raise ValueError("Must run .fit() first.")
        
        k = self.n_params
        L = self.log_likelihood_
        return 2 * k - 2 * L
        
    def get_bic(self):
        """
        Returns the Bayesian Information Criterion (BIC).
        Also for comparing models. Lower is better.
        
        BIC = k*log(n) - 2*L
        k = number of parameters
        n = number of observations
        L = log-likelihood
        """
        if self.log_likelihood_ is None:
            raise ValueError("Must run .fit() first.")
        
        k = self.n_params
        L = self.log_likelihood_
        n = self.n_obs
        return k * np.log(n) - 2 * L

    def __repr__(self):
        """Simple string representation."""
        if self.params:
            return f"{self.__class__.__name__}(params={self.params})"
        return f"{self.__class__.__name__}(not_fitted)"
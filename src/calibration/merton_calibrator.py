import numpy as np
import pandas as pd
from .base_calibrator import Calibrator
from .gbm_calibrator import GbmCalibrator 
from scipy.optimize import minimize
from scipy.stats import norm, poisson

class MertonCalibrator(Calibrator):
    """
    Calibrates the 5 parameters for a Merton Jump-Diffusion Model
    using Maximum Likelihood Estimation (MLE).
    """
    
    def __init__(self, dt=1/252, k_max=10):
        """
        Args:
            dt (float): The time step in years (default 1/252 for daily).
            k_max (int): How many jumps to sum over in the likelihood.
        """
        super().__init__()
        self.dt = dt
        self.k_max = k_max
        
    def fit(self, data):
        """
        Calibrates the 5 Merton parameters using MLE.
        """
        if not isinstance(data, pd.Series) or data.empty:
            raise ValueError("Data must be a non-empty pandas Series.")
        
        S0 = data.iloc[0]
        
        # log returns are what we're fitting to
        log_returns = np.log(data / data.shift(1)).dropna()
        
        # Store for AIC/BIC
        self.n_obs = len(log_returns)
        
        # --- Objective function to minimize ---
        # (This is the Negative Log-Likelihood)
        # We define it inside fit() so it can use log_returns
        def merton_neg_log_likelihood(params):
            
            # 1. Unpack the 5 parameters
            mu, sigma, lambda_j, mu_j, sigma_j = params
            
            # --- (THE FIX IS HERE) ---
            # We must use the *exact same* drift logic as the simulator
            
            # 1a. Calculate kappa (jump drift component)
            # Add a small epsilon to avoid issues if sigma_j is 0
            kappa = np.exp(mu_j + 0.5 * (sigma_j**2 + 1e-10)) - 1
            
            # 1b. Calculate the drift of the GBM part
            diffusion_drift = (mu - lambda_j * kappa - 0.5 * sigma**2) * self.dt
            # --- (END FIX) ---
            
            # 2. This is the math part (mixture model)
            # We sum the probabilities from k=0 to k=k_max jumps
            k_range = np.arange(self.k_max + 1)
            
            # Probability of k jumps
            poisson_probs = poisson.pmf(k_range, lambda_j * self.dt)
            
            # Mean and StdDev *given* k jumps
            mean_k = diffusion_drift + k_range * mu_j
            var_k = (sigma**2 * self.dt) + (k_range * sigma_j**2) + 1e-10
            std_k = np.sqrt(var_k)
            
            # 3. Vectorized calculation
            # Reshape returns to a column vector for broadcasting
            r_col = log_returns.values.reshape(-1, 1)
            
            # Calculate a (n_obs, k_max+1) matrix of probabilities
            normal_pdfs = norm.pdf(r_col, loc=mean_k, scale=std_k)
            
            # 4. Calculate the total likelihood for each return
            # (rows = returns, cols = k-jumps)
            weighted_pdfs = normal_pdfs * poisson_probs
            
            # Sum across the k-jumps (axis=1) to get the
            # final probability for each log return
            likelihoods = np.sum(weighted_pdfs, axis=1)
            
            # Add a small amount to avoid log(0)
            likelihoods[likelihoods <= 0] = 1e-10
            
            # 5. Get the total log-likelihood
            log_likelihoods = np.log(likelihoods)
            total_log_likelihood = np.sum(log_likelihoods)
            
            # If the optimizer picks bad params, return Inf
            if np.isnan(total_log_likelihood) or np.isinf(total_log_likelihood):
                return np.inf 
            
            # Return the *negative* LL (because we minimize)
            return -total_log_likelihood
        
        # --- End of the objective function ---
        
        # 2. Get a good Initial Guess (x0)
        # Use the GBM calibrator as a starting point
        gbm_cal = GbmCalibrator(dt=self.dt)
        gbm_params = gbm_cal.fit(data)
        
        mu_guess = gbm_params['mu']
        sigma_guess = gbm_params['sigma']
        
        # Guess the jump params
        lambda_j_guess = 1.0    # 1 jump per year
        mu_j_guess = 0.0      # jumps are neutral
        sigma_j_guess = 0.1   # 10% jump volatility
        
        x0 = [mu_guess, sigma_guess, lambda_j_guess, mu_j_guess, sigma_j_guess]
        
        # 3. Set Bounds (to keep params in a valid range)
        bounds = [
            (None, None),     # mu
            (1e-6, None),     # sigma (must be > 0)
            (1e-6, None),     # lambda_j (must be > 0)
            (None, None),     # mu_j
            (1e-6, None)      # sigma_j (must be > 0)
        ]
        
        # 4. Run the Optimizer
        # We suppress warnings here, as the optimizer
        # might hit bad values (e.g. log(0)) which is fine
        with np.errstate(all='ignore'):
            result = minimize(
                merton_neg_log_likelihood,
                x0,
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
        
        # If the optimizer fails, return empty params
        if not result.success:
            # Don't print a warning, just return nothing
            # This is cleaner for a Monte Carlo test
            self.params = {} 
            return self.params
        
        # 5. Store the results
        opt_params = result.x
        self.params = {
            'mu': opt_params[0],
            'sigma': opt_params[1],
            'lambda_j': opt_params[2],
            'mu_j': opt_params[3],
            'sigma_j': opt_params[4],
            'S0': S0
        }
        
        # Store the metrics for AIC/BIC
        self.log_likelihood_ = -result.fun # The NLL
        self.n_params = 5 # 5 params
        
        return self.params
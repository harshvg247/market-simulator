import numpy as np
from .base_process import StochasticProcess

# Inherit from our base class
class MertonJumpDiffusion(StochasticProcess):
    """
    Implements the Merton Jump-Diffusion Model.
    GBM + a compound Poisson process for jumps.
    
    Parameters:
    S0 (float): Initial stock price
    mu (float): Total expected drift (return)
    sigma (float): Volatility of the diffusion part
    lambda_j (float): Jump intensity (avg. number of jumps/year)
    mu_j (float): Mean of the log-jump size
    sigma_j (float): Volatility of the log-jump size
    """
    
    def _validate_params(self):
        """Check for all required GBM and Jump parameters."""
        required = ['S0', 'mu', 'sigma', 'lambda_j', 'mu_j', 'sigma_j']
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
        
        if self.params['sigma'] < 0 or self.params['sigma_j'] < 0:
            raise ValueError("Volatilities (sigma, sigma_j) cannot be negative.")
        if self.params['lambda_j'] < 0:
            raise ValueError("Jump intensity (lambda_j) cannot be negative.")
            
    def simulate(self, T, dt, n_paths=1):
        """
        Simulates Merton model paths.
        """
        
        # 1. Get all the parameters
        S0 = self.params['S0']
        mu = self.params['mu']
        sigma = self.params['sigma']
        lambda_j = self.params['lambda_j']
        mu_j = self.params['mu_j']
        sigma_j = self.params['sigma_j']
        
        # 2. Set up simulation arrays
        num_steps = int(T / dt) + 1
        
        paths = np.zeros((num_steps, n_paths))
        paths[0] = S0
        
        # 3. Calculate kappa (the jump drift adjustment)
        # This is E[J-1]
        kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        
        # 4. Pre-calculate the fixed GBM drift
        # (This is the total drift 'mu' *minus* the expected drift from jumps)
        gbm_drift = (mu - lambda_j * kappa - 0.5 * sigma**2) * dt
        
        # 5. Pre-calculate the GBM diffusion
        gbm_diffusion = sigma * np.sqrt(dt)
        
        # --- Generate all random numbers at once ---
        
        # Z1: Standard normals for the GBM part
        Z1 = np.random.normal(0, 1, size=(num_steps - 1, n_paths))
        
        # N: Poisson randoms for *how many* jumps in each step
        N = np.random.poisson(lambda_j * dt, size=(num_steps - 1, n_paths))
        
        # --- (THE FIX IS HERE) ---
        # We need the total jump size in each step.
        # If N jumps happen, the mean is N*mu_j
        # and the variance is N*sigma_j^2.
        
        # Mean of the total jump in one step
        jump_mean = N * mu_j

        # Standard deviation of the total jump in one step
        # (Var = N*sigma_j^2, so StdDev = sqrt(N)*sigma_j)
        # Add a tiny number to avoid sqrt(0)
        jump_std = np.sqrt(N * sigma_j**2 + 1e-10) 

        # Generate the total log-jump size from its correct distribution
        log_jump = np.random.normal(loc=jump_mean, scale=jump_std)
        # --- (END FIX) ---
        
        # 6. Run the simulation
        for t in range(1, num_steps):
            
            # The total return is the sum of the parts
            log_return_t = gbm_drift + gbm_diffusion * Z1[t-1] + log_jump[t-1]
            
            # Apply the return
            paths[t] = paths[t-1] * np.exp(log_return_t)
            
        return paths
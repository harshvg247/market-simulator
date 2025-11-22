import numpy as np
from .base_process import StochasticProcess

# Inherit from base class
class GeometricBrownianMotion(StochasticProcess):
    """
    Implements the Geometric Brownian Motion (GBM) process.
    
    SDE: dS_t = mu * S_t * dt + sigma * S_t * dW_t
    
    Using the log-normal solution:
    S_t = S_{t-1} * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
    
    Parameters:
    S0 (float): Initial stock price
    mu (float): Drift (expected return)
    sigma (float): Volatility
    """
    
    def _validate_params(self):
        """Check if S0, mu, and sigma are present and valid."""
        required = ['S0', 'mu', 'sigma']
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Volatility can't be negative
        if self.params['sigma'] < 0:
            raise ValueError("sigma cannot be negative.")
        
    def simulate(self, T, dt, n_paths=1):
        """
        Simulates GBM paths.
        """

        S0 = self.params['S0']
        mu = self.params['mu']
        sigma = self.params['sigma']

        num_steps = int(T / dt) + 1
        
        # Shape: (time, paths)
        paths = np.zeros((num_steps, n_paths))
        paths[0] = S0  # Set S0 for all paths
        
        # Precompute constants
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate all random numbers (Standard Normal)
        # (num_steps-1 because we don't need one for S0)
        Z = np.random.normal(0, 1, size=(num_steps - 1, n_paths))
        
        # Simulation loop
        for t in range(1, num_steps):
            paths[t] = paths[t-1] * np.exp(drift + diffusion * Z[t-1])
            
        return paths
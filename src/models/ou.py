import numpy as np
from .base_process import StochasticProcess

# Inherit from our base class
class OrnsteinUhlenbeck(StochasticProcess):
    """
    Implements the Ornstein-Uhlenbeck (OU) mean-reverting process.
    
    SDE: dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    
    Parameters:
    X0 (float): Initial value of the process.
    mu (float): The long-term mean (equilibrium level).
    theta (float): The speed of reversion.
    sigma (float): The volatility of the process.
    """
    
    def _validate_params(self):
        """Check for required parameters X0, mu, theta, and sigma."""
        required = ['X0', 'mu', 'theta', 'sigma']
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
        
        if self.params['sigma'] < 0:
            raise ValueError("sigma cannot be negative.")
        if self.params['theta'] < 0:
            raise ValueError("theta (reversion speed) cannot be negative.")
            
    def simulate(self, T, dt, n_paths=1):
        """
        Simulates OU paths using the Euler-Maruyama discretization.
        """
        
        X0 = self.params['X0']
        mu = self.params['mu']
        theta = self.params['theta']
        sigma = self.params['sigma']
        
        num_steps = int(T / dt) + 1
        
        paths = np.zeros((num_steps, n_paths))
        paths[0] = X0
        
        diffusion_term = sigma * np.sqrt(dt)
        
        # Generate all random numbers
        Z = np.random.normal(0, 1, size=(num_steps - 1, n_paths))
        
        # simulation
        for t in range(1, num_steps):
            # Get the previous value
            X_prev = paths[t-1]
            
            # Calculate the two parts of the SDE
            # a) The "drift" or "reversion" part
            reversion = theta * (mu - X_prev) * dt
            
            # b) The "diffusion" or "random" part
            diffusion = diffusion_term * Z[t-1]
            
            # Update the path
            paths[t] = X_prev + reversion + diffusion
            
        return paths
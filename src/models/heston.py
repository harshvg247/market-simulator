import numpy as np
from .base_process import StochasticProcess

# Inherit from our base class
class HestonModel(StochasticProcess):
    """
    Implements the Heston Stochastic Volatility Model.
    
    This model has two correlated SDEs:
    1. Asset Price (S_t): 
       dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_S
    2. Variance (v_t):   
       dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_v
       
    And Corr(dW_S, dW_v) = rho
    
    Parameters:
    S0 (float): Initial stock price.
    mu (float): The drift coefficient (expected return).
    v0 (float): Initial variance.
    kappa (float): Speed of mean reversion for variance.
    theta (float): The long-term mean variance.
    xi (float): Volatility of variance ("vol of vol").
    rho (float): Correlation between asset and variance processes.
    """
    
    def _validate_params(self):
        """Check for all 7 required Heston parameters."""
        required = ['S0', 'mu', 'v0', 'kappa', 'theta', 'xi', 'rho']
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
        
        if self.params['rho'] < -1 or self.params['rho'] > 1:
            raise ValueError("Correlation (rho) must be between -1 and 1.")
        
        if (self.params['v0'] < 0 or self.params['kappa'] < 0 or 
            self.params['theta'] < 0 or self.params['xi'] < 0):
            raise ValueError("v0, kappa, theta, and xi must be non-negative.")
            
        # Optional: Check Feller condition (2*kappa*theta > xi^2)
        # This ensures variance never hits 0.
        # We handle this with reflection, so it's not a strict requirement.

    def simulate(self, T, dt, n_paths=1):
        """
        Simulates Heston paths using a full truncation (reflection) scheme
        to prevent negative variance.
        """
        S0 = self.params['S0']
        mu = self.params['mu']
        v0 = self.params['v0']
        kappa = self.params['kappa']
        theta = self.params['theta']
        xi = self.params['xi']
        rho = self.params['rho']
        
        num_steps = int(T / dt) + 1
        
        paths = np.zeros((num_steps, n_paths))
        vars = np.zeros((num_steps, n_paths))
        paths[0] = S0
        vars[0] = v0

        Z_S_ind = np.random.normal(0, 1, size=(num_steps - 1, n_paths))
        Z_v_ind = np.random.normal(0, 1, size=(num_steps - 1, n_paths))
        
        # 5. Creating the correlated random numbers
        # W_S = Z_S_ind
        # W_v = rho * Z_S_ind + sqrt(1 - rho^2) * Z_v_ind
        W_S = Z_S_ind
        W_v = rho * Z_S_ind + np.sqrt(1 - rho**2) * Z_v_ind
        
        sqrt_dt = np.sqrt(dt)
        
        # simulation
        for t in range(1, num_steps):
            # --- First, update the variance ---
            v_prev = vars[t-1]
            
            # Prevent negative variance (Full Truncation)
            v_prev_safe = np.maximum(v_prev, 0)
            
            # Discretization for v_t
            drift_v = kappa * (theta - v_prev_safe) * dt
            diffusion_v = xi * np.sqrt(v_prev_safe) * W_v[t-1] * sqrt_dt
            
            vars[t] = np.maximum(v_prev + drift_v + diffusion_v, 0)
            
            # --- Second, update the asset price ---
            S_prev = paths[t-1]
            v_t_sqrt = np.sqrt(v_prev_safe) # use the *previous* variance
            
            # Discretization for S_t
            drift_S = (mu - 0.5 * v_prev_safe) * dt
            diffusion_S = v_t_sqrt * W_S[t-1] * sqrt_dt
            
            paths[t] = S_prev * np.exp(drift_S + diffusion_S)
  
        return paths, vars # Returning both for validation
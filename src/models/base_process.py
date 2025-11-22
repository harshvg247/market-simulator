import numpy as np
from abc import ABC, abstractmethod

# ABC is the Abstract Base Class that ensures all stochastic models
# follow a common structure (like having simulate(), etc.)
class StochasticProcess(ABC):
    """
    Base abstract class for any stochastic process model.
    Every specific model (like GBM, Vasicek, CIR, etc.)
    should inherit from this and define its own logic.
    """
    
    def __init__(self, **params):
        """
        Initializes the model with given parameters.
        """
        self.params = params
        self._validate_params() # check if we have all params
        
    @abstractmethod
    def _validate_params(self):
        """
        Checks if all needed parameters were passed in.
        Should be implemented by each child class.
        """
        pass
    
    @abstractmethod
    def simulate(self, T, dt, n_paths=1):
        """
        The main simulation function - Must be implemented by each child class.
        Args:
            T (float): Total time horizon.
            dt (float): Time step size (e.g., 1/252 for daily).
            n_paths (int): Number of paths to simulate.
            
        Returns:
            A 2D numpy array: (num_steps, n_paths)
        """
        pass
        
    def get_params(self):
        """Return the model parameters as a dictionary."""
        return self.params

    def __repr__(self):
        """Simple string representation for printing the model."""
        return f"{self.__class__.__name__}({self.params})"
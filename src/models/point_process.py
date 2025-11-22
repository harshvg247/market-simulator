import numpy as np
from abc import ABC, abstractmethod

class PointProcess(ABC):
    """
    Abstract Base Class for point process models (e.g., Hawkes, Poisson).
    
    This class defines the "contract" for models that simulate the
    *times* of discrete events, rather than a continuous path.
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
    def simulate(self, T, n_paths=1):
        """
        The main simulation function.
        
        Args:
            T (float): Total time horizon.
            n_paths (int): Number of paths to simulate.
            
        Returns:
            A list of length 'n_paths', where each element
            is a 1D numpy array of event times.
        """
        pass
        
    def get_params(self):
        """Helper to get the model's parameters."""
        return self.params

    def __repr__(self):
        """Simple string representation for printing the model."""
        return f"{self.__class__.__name__}({self.params})"
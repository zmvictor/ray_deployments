from abc import ABC, abstractmethod

"""
BaseModel is an abstract class that defines the interface for all models.
"""
class BaseModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def model_params(self) -> dict:
        # Return a dictionary of model parameters
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        # The serving system will call this before routing requests to the model instance
        pass
    
    @abstractmethod
    def eval(self, *args, **kwargs):
        # The serving system will call this to evaluate the model on a request
        pass

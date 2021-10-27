import torch
from abc import ABC, abstractmethod


class INet(ABC):
    """interface for the value function network
    """
    @abstractmethod
    def compute_values(self, query: torch.Tensor) -> torch.Tensor:
        """passes a query tensor [batch, query_dim] to the net and returns
        expected values [batch, belief_size]
        """
        pass

    @abstractmethod
    def add_training_example(
        self,
        query: torch.Tensor,
        values: torch.Tensor
    ) -> None:
        pass

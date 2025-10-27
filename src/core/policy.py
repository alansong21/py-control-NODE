import torch
import torch.nn as nn
from typing import Tuple, Optional

class NeuralPolicy(nn.Module):
    """Neural policy matching Julia FastChain architecture"""
     
    def __init__(
        self,
        input_dim: list[int],  # input curr state + goal state
        hidden_dims: Tuple[int, ...] = (12, 12),
        output_dim: int = 1,  # dynamics output: , control output: int
        activation: str = "tanh",
        control_bounds: Optional[Tuple[float, float]] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.control_bounds = control_bounds

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights (Xavier initialization)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional parameter override"""
        if params is not None:
            # Use provided parameters instead of learned parameters
            return self._forward_with_params(x, params)
        
        A = torch.tensor([[0,1,0],[0,0,1],[0,0,0]])
        B = torch.tensor([[0,0,1]])
        u = torch.tensor([0,0,self.network(x)])
        
        # output = self.network(x)
        output = A @ x + B @ u

        # Apply control constraints if provided
        if self.control_bounds is not None:
            lower, upper = self.control_bounds
            # Scale sigmoid output to [lower, upper]
            output = torch.sigmoid(output) * (upper - lower) + lower

        return output
    
    def _forward_with_params(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Forward pass using explicit parameters (for optimization)"""

        # Optimization approach will be implemented later
        pass
       

       
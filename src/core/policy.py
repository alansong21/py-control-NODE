import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Tuple, Optional, List, Dict

class NeuralPolicy(nn.Module):
    """Neural policy matching Julia FastChain architecture"""
     
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (12, 12),
        output_dim: int = 1,
        activation: str = "tanh",
        control_bounds: Optional[Tuple[float, float]] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.control_bounds = control_bounds

        # Build network layers
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unsupported activation function: '{activation}'")
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights (Xavier initialization)
        self._initialize_weights()
        self._parameter_spec = self._collect_parameter_spec()

    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _collect_parameter_spec(self) -> List[Dict[str, Tuple[int, ...]]]:
        """Record weight/bias shapes to unpack flattened parameter vectors."""
        spec: List[Dict[str, Tuple[int, ...]]] = []
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                spec.append(
                    {
                        "weight_shape": tuple(layer.weight.shape),
                        "weight_numel": layer.weight.numel(),
                        "bias_shape": tuple(layer.bias.shape),
                        "bias_numel": layer.bias.numel(),
                    }
                )
        return spec
    
    def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional parameter override"""
        if params is not None:
            # Use provided parameters instead of learned parameters
            return self._forward_with_params(x, params)
        
        output = self.network(x if x.dim() > 1 else x.unsqueeze(0))

        # Apply control constraints if provided
        if self.control_bounds is not None:
            lower, upper = self.control_bounds
            # Scale sigmoid output to [lower, upper]
            output = torch.sigmoid(output) * (upper - lower) + lower

        return output.squeeze(0) if x.dim() == 1 else output
    
    def _forward_with_params(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Forward pass using explicit parameters (for optimization)"""
        if params.dim() != 1:
            raise ValueError("Parameter override must be a 1-D tensor.")
        
        pointer = 0
        output = x if x.dim() > 1 else x.unsqueeze(0)
        spec_index = 0
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                layer_spec = self._parameter_spec[spec_index]
                spec_index += 1

                weight_numel = layer_spec["weight_numel"]
                bias_numel = layer_spec["bias_numel"]

                weight = params[pointer : pointer + weight_numel].view(layer_spec["weight_shape"])
                pointer += weight_numel
                bias = params[pointer : pointer + bias_numel].view(layer_spec["bias_shape"])
                pointer += bias_numel

                output = output @ weight.t() + bias
            else:
                output = layer(output)

        if pointer != params.numel():
            raise ValueError("Parameter vector size does not match network architecture.")
        
        if self.control_bounds is not None:
            lower, upper = self.control_bounds
            output = torch.sigmoid(output) * (upper - lower) + lower

        return output.squeeze(0) if x.dim() == 1 else output
    
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(spec["weight_numel"] + spec["bias_numel"] for spec in self._parameter_spec)
    
    def get_parameter_vector(self) -> torch.Tensor:
        """Flatten current weights into a single vector."""
        return parameters_to_vector(self.parameters()).detach().clone()
    
    def load_parameter_vector(self, vector: torch.Tensor) -> None:
        """Load weights from a flattened parameter vector."""
        if vector.numel() != self.num_parameters():
            raise ValueError("Vector length does not match number of parameters.")
        device = next(self.parameters()).device
        vector_to_parameters(vector.to(device), self.parameters())


       
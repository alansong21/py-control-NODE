import torch
from typing import Tuple, Optional

class TripleIntegrator:
    """Triple integrator system"""

    def __init__(self, mu: float = 1.0):
        self.mu = mu
    
    def __call__(
            self,
            x: torch.Tensor,
            params: torch.Tensor,
            t: torch.Tensor,
            controller: torch.nn.Module,
            input_type: str = "state"
    ) -> torch.Tensor:
        """
        Triple integrator dynamics:
        x' = v
        v' = a
        a' = u (control)
        """
        # Extract state components
        pos, vel, acc = x[0], x[1], x[2]

        # Get control input from neural policy
        if input_type == "state":
            u = controller(x, params)[0] # Control based on state
        elif input_type == "time":
            u = controller(t, params)[0] # Control based on time
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
    
        # Dynamics: x' = v, v' = a, a' = u
        dxdt = torch.stack([vel, acc, u])

        return dxdt
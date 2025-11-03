import torch
from typing import Tuple, Optional

class TripleIntegrator:
    """Triple integrator system"""

    def __init__(self, mu: float = 1.0):
        self.mu = mu
    
    def __call__(
            self,
            x: torch.Tensor,
            params: Optional[torch.Tensor],
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
            policy_input = x
        elif input_type == "time":
            policy_input = torch.as_tensor(
                [t], dtype=x.dtype, device=x.device
            )   # treat time as 1D input
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        control = controller(policy_input, params) if params is not None else controller(policy_input)
        u = control[0]
    
        # Dynamics: x' = v, v' = a, a' = u
        dxdt = torch.stack([vel, acc, u])

        return dxdt
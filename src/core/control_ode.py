from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Union
import torch
import numpy as np

@dataclass
class ControlODEConfig:
    """Configuration for ControlODE system"""
    tspan: Tuple[float, float]
    dt: float = 0.01
    integrator: str = "dopri5" # torchdiffeq integrator
    sensealg: str = "autograd" # sensitivity algorithm
    input_type: str = "state"  # "state" or "time"

@dataclass
class TrainingConfig:
    """Training Configuration"""
    max_iterations: int = 1000
    optimizer: str = "adam"
    learning_rate: float = 1e-2
    tolerance: float = 1e-6
    constraint_method: str = "barrier" # "barrier" or "penalty"

class ControlODE:
    """Main ControlODE class -- equivalent to Julia ControlODE struct"""

    def __init__(
            self,
            policy: torch.nn.Module,
            system: Callable,
            initial_state: torch.Tensor,
            config: ControlODEConfig,
    ):
        self.policy = policy
        self.system = system
        self.u0 = initial_state
        self.config = config
        self.tsteps = torch.linspace(
            config.tspan[0], config.tspan[1],
            int((config.tspan[1] - config.tspan[0]) / config.dt) + 1
        )
    
    def solve(self, params: torch.Tensor) -> torch.Tensor:
        """Solve ODE with given parameters"""
        from torchdiffeq import odeint

        def ode_func(t, x):
            return self.system(x, params, t, self.policy)
        
        return odeint(ode_func, self.u0, self.tsteps)
    
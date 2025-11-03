import torch
from typing import Dict, Any, Callable, Optional
from ..core.control_ode import ControlODE

class CostFunction:
    """Base class for cost functions"""

    def __init__(self, control_ode: ControlODE, config: Dict[str, Any] = None):
        self.control_ode = control_ode
        self.config = config or {}

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        """Compute cost for given parameters"""
        raise NotImplementedError
    
class TripleIntegratorCost(CostFunction):
    """Cost function for triple integrator"""

    def __init__(self, control_ode: ControlODE, config: Dict[str, Any] = None):
        super().__init__(control_ode, config)
        self.dt = control_ode.config.dt

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        """Compute objective: integral(x1^2 + 0.001x2^2 + 0.001x3^2, dt)"""
        # Solve ODE with given parameters
        trajectory = self.control_ode.solve(params)

        objective = torch.zeros((), dtype=trajectory.dtype, device=trajectory.device)

        # Integrate cost over trajectory
        for i in range(trajectory.shape[0]):
            state = trajectory[i]
            # State cost: position^2 + 0.001*velocity^2 _ 0.001*acceleration^2
            state_cost = state[0]**2 + 0.001 * state[1]**2 + 0.001 * state[2]**2
            objective = objective + state_cost

        return objective * self.dt
    
class ConstrainedCost(CostFunction):
    """Cost function with constraints using barrier methods"""

    def __init__(
        self,
        control_ode: ControlODE,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        constraints: Dict[str, Any],
        config: Dict[str, Any] = None
    ):
        super().__init__(control_ode, config)
        self.objective_fn = objective_fn
        self.constraints = constraints
        self.alpha = config.get('alpha', 1.0) # Penalty weight
        self.delta = config.get('delta', 0.1) # Barrier relaxation
        self.rho = config.get('rho', 0.0)     # Regularization
        self.epsilon = self.config.get("epsilon", 1e-6)

    def __call__(
            self, 
            params: torch.Tensor,
            alpha: Optional[float] = None,
            delta: Optional[float] = None,
            rho: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute total cost with constraints"""
        trajectory = self.control_ode.solve(params)

        # Objective cost
        objective = self.objective_fn(trajectory)
        alpha_val = float(alpha) if alpha is not None else self.alpha
        delta_val = float(delta) if delta is not None else self.delta
        rho_val = float(rho) if rho is not None else self.rho

        # State constraints using relaxed log barriers
        state_penalty = self._compute_state_penalty(trajectory, alpha_val, delta_val)

        # Control penalty
        control_penalty = self._compute_control_penalty(trajectory, params)

        # Regularization
        regularization = rho_val * torch.sum(params**2)

        return {
            'objective': objective,
            'state_penalty': state_penalty,
            'control_penalty': control_penalty,
            'regularization': regularization,
            'total': objective + state_penalty + control_penalty + regularization
        }

    def _compute_state_penalty(
            self, 
            trajectory: torch.Tensor,
            alpha: float,
            delta: float
    ) -> torch.Tensor:
        """Compute state constraint penalties using relaxed log barriers"""
        penalty = torch.zeros((), dtype=trajectory.dtype, device=trajectory.device)
        dt = self.control_ode.config.dt

        for constraint_info in self.constraints.values():
            if constraint_info['type'] != 'state':
                continue

            var_idx = constraint_info['variable']
            bounds = constraint_info['bounds']
            values = trajectory[:, var_idx]

            penalty_terms = self._relaxed_log_barrier(values, bounds[0], bounds[1], delta)
            penalty = penalty + torch.sum(penalty_terms)
        
        return alpha * dt * penalty
    
    def _relaxed_log_barrier(
            self,
            x: torch.Tensor,
            lower: float,
            upper: float,
            delta: float,
    ) -> torch.Tensor:
        """Relaxed log barrier function"""
        # Barrier for lower bound: x > lower
        lower_barrier = self._single_barrier(x - lower, delta)

        # Barrier for upper bound: x < upper
        upper_barrier = self._single_barrier(upper - x, delta)

        return lower_barrier + upper_barrier
    
    def _single_barrier(self, z: torch.Tensor, delta: float) -> torch.Tensor:
        """Single-sided relaxed log barrier"""
        delta_tensor = torch.tensor(delta, dtype=z.dtype, device=z.device)
        clamped = torch.clamp(z, min=self.epsilon)
        # Exponential relaxation for z <= delta
        relaxation = torch.exp(1.0 - z / delta) - 1.0 - torch.log(delta_tensor)
        # Log barrier for z > delta
        log_barrier = -torch.log(clamped)

        return torch.where(z > delta_tensor, log_barrier, relaxation)
    
    def _compute_control_penalty(self, trajectory: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute control penalty (placeholder for now)"""
        return torch.zeros((), dtype=trajectory.dtype, device=trajectory.device)
    

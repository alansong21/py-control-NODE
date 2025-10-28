import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from typing import Dict, Any, Tuple, Callable
from dataclasses import dataclass

try:
    from tqdm.auto import trange
except ImportError:  # pragma: no cover - optional dependency
    trange = None

@dataclass
class BarrierConfig:
    """Configuration for barrier method"""
    alpha0: float = 1.0
    delta0: float = 10.0
    max_iterations: int = 100
    alpha_increase_factor: float = 1.5
    delta_decrease_factor: float = 0.9
    penalty_ratio_bounds: Tuple[float, float] = (0.1, 10.0)

class BarrierMethod:
    """Fiacco-McCormick barrier method implementation"""

    def __init__(self, config: BarrierConfig = None):
        self.config = config or BarrierConfig()

    def tune_barrier_parameters(
        self,
        cost_fn: Callable,
        params: torch.Tensor,
        alpha: float,
        delta: float,
        rho: float,
        max_iters: int = 10
    ) -> Tuple[float, float]:
        """Tune barrier parameters based on penalty ratios"""

        for _ in range(max_iters):
            # Evaluate current barrier parameters
            costs = cost_fn(params, alpha=alpha, delta=delta, rho=rho)

            # Check penalty ratio
            state_penalty = costs["state_penalty"].detach()
            objective = costs["objective"].detach()

            if not torch.isfinite(state_penalty):
                return alpha, delta
            
            other_costs_size = torch.abs(objective) + 1e-12
            penalty_ratio = torch.abs(state_penalty) / other_costs_size
            ratio_value = penalty_ratio.item()

            if ratio_value > self.config.penalty_ratio_bounds[1]:
                delta *= self.config.alpha_increase_factor
            elif ratio_value < self.config.penalty_ratio_bounds[0]:
                delta *= self.config.delta_decrease_factor
            else:
                return alpha, delta
            
        return alpha, delta
    
    def constrained_optimization(
        self,
        cost_fn: Callable,
        initial_params: torch.Tensor,
        optimizer_config: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Main constrained optimization loop"""
        
        params = initial_params.clone()
        alpha = self.config.alpha0
        delta = self.config.delta0
        rho = optimizer_config.get('rho', 0.0)

        alpha_progression = [alpha]
        delta_progression = [delta]
        last_iteration = 0

        show_progress = bool(optimizer_config.get("show_progress", False) and trange is not None)
        outer_desc = optimizer_config.get("outer_progress_desc", "Barrier iterations")
        outer_iterator = (
            trange(self.config.max_iterations, desc=outer_desc)
            if show_progress
            else range(self.config.max_iterations)
        )

        for iteration in outer_iterator:
            last_iteration = iteration
            # Tune barrier parameters
            alpha, delta = self.tune_barrier_parameters(
                cost_fn, params, alpha, delta, rho
            )

            # Optimize with current barrier parameters
            def loss_fn(p: torch.Tensor) -> torch.Tensor:
                costs = cost_fn(p, alpha=alpha, delta=delta, rho=rho)
                return costs['total']
        
            # Use optimizer to minimize loss
            params = self._optimize_step(loss_fn, params, optimizer_config)

            # Store progression
            alpha_progression.append(alpha)
            delta_progression.append(delta)

            if show_progress:
                outer_iterator.set_postfix(alpha=alpha, delta=delta)

            # Check convergence
            if self._check_convergence(alpha_progression, delta_progression):
                break

        if show_progress:
            outer_iterator.close()
        
        return params, {
            'alpha_progression': alpha_progression,
            'delta_progression': delta_progression,
            'iterations': last_iteration + 1
        }

    def _optimize_step(
            self,
            loss_fn: Callable,
            params: torch.Tensor,
            config: Dict[str, Any]
    ) -> torch.Tensor:
        """Single optimization step -- placeholder for optimizer integration"""
        lr = config.get("learning_rate", 1e-2)
        steps = config.get("max_steps", 200)
        optimizer_name = config.get("optimizer", "adam").lower()
        betas = config.get("betas", (0.9, 0.999))
        grad_clip = config.get("grad_clip")

        theta = torch.nn.Parameter(params.detach().clone())
        show_progress = bool(config.get("show_progress", False) and trange is not None)
        inner_desc = config.get("progress_desc", "Inner optimization")
        iterator = (
            trange(steps, desc=inner_desc, leave=False)
            if show_progress
            else range(steps)
        )

        if optimizer_name == "sgd":
            optimizer = optim.SGD([theta], lr=lr)
        else:
            optimizer = optim.Adam([theta], lr=lr, betas=betas)

        for _ in iterator:
            optimizer.zero_grad()
            loss = loss_fn(theta)
            if not torch.isfinite(loss):
                break
            loss.backward()
            if grad_clip is not None:
                clip_grad_norm_([theta], grad_clip)
            optimizer.step()
            if show_progress:
                iterator.set_postfix(loss=float(loss.detach()))

        if show_progress:
            iterator.close()

        return theta.detach()
    
    def _check_convergence(
            self, alpha_prog: list,
            delta_prog: list
    ) -> bool:
        """Check if optimization has converged"""
        # Simple convergence check
        if len(alpha_prog) < 3:
            return False

        alpha_change = abs(alpha_prog[-1] - alpha_prog[-2])
        delta_change = abs(delta_prog[-1] - delta_prog[-2])

        return alpha_change < 1e-6 and delta_change < 1e-6

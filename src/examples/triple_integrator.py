import argparse 
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import torch
import matplotlib.pyplot as plt


try:
    from tqdm.auto import trange
except ImportError:  # pragma: no cover - optional dependency
    trange = None

from src.core.policy import NeuralPolicy
from src.core.control_ode import ControlODE, ControlODEConfig
from src.core.system import TripleIntegrator
from src.utils.core_functions import ConstrainedCost
from src.optimization.barriers import BarrierMethod, BarrierConfig
from src.ruckig_generator import ruckig_generator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "triple_integrator.json"

# Loads config from given path
def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level object.")
    return data

# Converts an Iterable to a Tuple
def _to_tuple(seq: Iterable[Any]) -> Tuple[Any, ...]:
    return tuple(seq)

# Computes the total cost penalty over a given trajectory
def objective_from_trajectory(
        trajectory: torch.Tensor,
        weights: Dict[str, float],
        dt: float,
) -> torch.Tensor:
    # Integrate state penalties over time for the trajectory, and return sum
    position_cost = weights.get("position", 1.0) * trajectory[:, 0] ** 2
    velocity_cost = weights.get("velocity", 1e-3) * trajectory[:, 1] ** 2
    acceleration_cost = weights.get("acceleration", 1e-3) * trajectory[:, 2] ** 2
    return dt * torch.sum(position_cost + velocity_cost + acceleration_cost)

def objective_from_control_error(
        trajectory: torch.Tensor,  
        dt: float,
        ground_truth: torch.Tensor ) -> torch.Tensor:
    
    trajectory_jerk = trajectory[:, 3]
    ground_truth_jerk = ground_truth[:, 4]
    
    print("traj shape", trajectory_jerk.shape)  # 251 right now
    print("ground_truth shape", ground_truth_jerk.shape)  # 280 right now
    
    
    return dt * (trajectory_jerk - ground_truth_jerk) ** 2

# Optimizes a given cost function without state or control constraints, using Adam
# Returns parameters that minimize the cost function without constraints
def optimize_unconstrained(
        cost_fn: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        steps: int,
        lr: float,
        show_progress: bool = False,
) -> torch.Tensor:
    theta = torch.nn.Parameter(params.clone().detach())
    optimizer = torch.optim.Adam([theta], lr=lr)

    iterator = (
        trange(steps, desc="Unconstrained training", leave=False)
        if show_progress and trange is not None
        else range(steps)
    )

    for _ in iterator:
        optimizer.zero_grad()
        loss = cost_fn(theta)
        loss.backward()
        optimizer.step()

        if show_progress and trange is not None:
            iterator.set_postfix(loss=float(loss.detach()))

    if show_progress and trange is not None:
        iterator.close()

    return theta.detach()

# Main triple integrator policy training loop
def run_triple_integrator(config: Dict[str, Any], ground_truth_traj) -> Dict[str, Any]:
    # Get configs
    system_cfg = config.get("system", {})
    policy_cfg = config.get("policy", {})
    objective_cfg = config.get("objective", {})
    constraints_cfg = config.get("constraints", {})
    training_cfg = config.get("training", {})

    initial_state = torch.tensor(system_cfg.get("initial_state", [0.0, 0.0, 0.0]), dtype=torch.float32)
    tspan = tuple(system_cfg.get("tspan", (0.0, 1.0)))
    dt = float(system_cfg.get("dt", 0.01))
    mu = float(system_cfg.get("mu", 1.0))

    input_dim = int(policy_cfg.get("input_dim", len(initial_state)))
    hidden_dims = _to_tuple(int(x) for x in policy_cfg.get("hidden_dims", [12, 12]))
    output_dim = int(policy_cfg.get("output_dim", 1))
    activation = policy_cfg.get("activation", "tanh")
    control_bounds = tuple(float(x) for x in policy_cfg.get("control_bounds", [-10.0, 10.0]))

    # Generate ControlODE config
    ode_config = ControlODEConfig(tspan=tspan, dt=dt)

    # Initialize policy
    policy = NeuralPolicy(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
        control_bounds=control_bounds,
    )

    # Set system to TripleIntegrator
    system = TripleIntegrator(mu=mu)

    # Initialize ODE
    ode = ControlODE(policy, system, initial_state, ode_config)

    # Fetch objective weights for cost
    objective_weights = {
        "position": float(objective_cfg.get("position", 1.0)),
        "velocity": float(objective_cfg.get("velocity", 1e-3)),
        "acceleration": float(objective_cfg.get("acceleration", 5e-3)),
    }

    # Fetch training configs
    unconstrained_steps = int(training_cfg.get("unconstrained_steps", 400))
    unconstrained_lr = float(training_cfg.get("unconstrained_lr", 5e-3))
    show_progress = bool(training_cfg.get("show_progress", True))

    # Compute the cost given a set of parameters
    def unconstrained_loss(params: torch.Tensor) -> torch.Tensor:
        trajectory = ode.solve(params)
        controls = ode.policy(trajectory, params)
        
        trajectory_extended = torch.cat((trajectory, controls), dim=1)
        # print("traj shape:", trajectory.shape)
        
        # return objective_from_trajectory(trajectory, objective_weights, dt, ground_truth_traj)
        return objective_from_control_error(trajectory_extended, dt, ground_truth_traj)

    # First, seed parameters by optimizing without constraints
    base_params = policy.get_parameter_vector()
    unconstrained_params = optimize_unconstrained(
        unconstrained_loss,
        base_params,
        unconstrained_steps,
        unconstrained_lr,
        show_progress=show_progress,
    )

    # Fetch constraint/bound configs
    velocity_bounds = tuple(float(x) for x in constraints_cfg.get("velocity", [-1.0, 1.0]))
    acceleration_bounds = tuple(float(x) for x in constraints_cfg.get("acceleration", [-3.0, 3.0]))

    constraints = {
        "velocity": {"type": "state", "variable": 1, "bounds": velocity_bounds},
        "acceleration": {"type": "state", "variable": 2, "bounds": acceleration_bounds},
    }

    # Fetch barrier method configs
    barrier_alpha0 = float(training_cfg.get("barrier_alpha0", 1.0))
    barrier_delta0 = float(training_cfg.get("barrier_delta0", 10.0))
    barrier_max_iterations = int(training_cfg.get("barrier_max_iterations", 25))
    penalty_ratio_bounds = tuple(float(x) for x in training_cfg.get("penalty_ratio_bounds", [0.0, 1.0]))
    rho = float(training_cfg.get("rho", 0.0))
    barrier_learning_rate = float(training_cfg.get("barrier_learning_rate", 5e-3))
    barrier_steps = int(training_cfg.get("barrier_steps", 200))

    # Initialize constrained cost and barrier method objects
    constrained_cost = ConstrainedCost(
        ode,
        # objective_fn=lambda traj: objective_from_trajectory(traj, objective_weights, dt, ground_truth_traj),
        objective_fn=lambda traj: objective_from_control_error(traj, dt, ground_truth_traj),
        constraints=constraints,
        config={"alpha": barrier_alpha0, "delta": barrier_delta0, "rho": rho},
    )

    barrier = BarrierMethod(
        BarrierConfig(
            alpha0=barrier_alpha0,
            delta0=barrier_delta0,
            max_iterations=barrier_max_iterations,
            penalty_ratio_bounds=penalty_ratio_bounds,
        )
    )

    # Using the seeded parameters, optimize policy subject to state and control constraints by using barrier method 
    constrained_params, history = barrier.constrained_optimization(
        constrained_cost,
        unconstrained_params,
        {
            "learning_rate": barrier_learning_rate,
            "max_steps": barrier_steps,
            "rho": rho,
            "show_progress": show_progress,
            "progress_desc": "Barrier inner optimization",
            "outer_progress_desc": "Barrier iterations",
        },
    )

    return {
        "ode": ode,
        "unconstrained_params": unconstrained_params,
        "constrained_params": constrained_params,
        "training_history": history,
        "config": config,
    }

# Plot trajectory given optimized policy and parameters
def plot_trajectory(ode: ControlODE, params: torch.Tensor, config: Dict[str, Any]) -> None:
    """Simulate the system with given params and plot states + control."""
    constraints_cfg = config.get("constraints", {})
    velocity_bounds = tuple(float(x) for x in constraints_cfg.get("velocity", [-1.0, 1.0]))
    acceleration_bounds = tuple(float(x) for x in constraints_cfg.get("acceleration", [-3.0, 3.0]))

    ode.policy.load_parameter_vector(params)
    trajectory = ode.solve(params).detach()
    times = ode.tsteps.detach()

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    labels = ["Position", "Velocity", "Acceleration"]

    for idx, label in enumerate(labels):
        axes[idx].plot(times.cpu().numpy(), trajectory[:, idx].cpu().numpy(), label=label)
        axes[idx].set_ylabel(label)
        axes[idx].grid(True)
        if label == "Velocity":
            axes[idx].axhline(y=velocity_bounds[0], color="r", linestyle="--", alpha=0.6)
            axes[idx].axhline(y=velocity_bounds[1], color="r", linestyle="--", alpha=0.6)
        if label == "Acceleration":
            axes[idx].axhline(y=acceleration_bounds[0], color="r", linestyle="--", alpha=0.6)
            axes[idx].axhline(y=acceleration_bounds[1], color="r", linestyle="--", alpha=0.6)
        axes[idx].legend(loc="upper right")

    controls = ode.policy(trajectory, params).squeeze(-1).detach()
    axes[-1].plot(times.cpu().numpy(), controls.cpu().numpy(), label="control u(t)")
    axes[-1].set_ylabel("Control")
    axes[-1].set_xlabel("Time")
    axes[-1].grid(True)
    axes[-1].legend(loc="upper right")

    fig.suptitle("Triple Integrator Trajectory")
    fig.tight_layout()
    fig.savefig("triple_integrator_trajectory.png", dpi=300)

# Parse arguments 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run triple integrator example.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config file.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting the trajectory.",
    )
    return parser.parse_args()

# Main function
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    rg = ruckig_generator(config)
    inp = rg.build_ruckig_input()
    
    # Run Ruckig
    traj, t = rg.run_ruckig(inp)
    
    traj = torch.tensor(traj)

    print("traj:\n", traj)
    print()
    
    results = run_triple_integrator(config, traj)
    
    print("Barrier iterations:", results["training_history"]["iterations"])
    if not args.skip_plot:
        plot_trajectory(results["ode"], results["constrained_params"], config)


if __name__ == "__main__":
    main()

import torch
import matplotlib.pyplot as plt

from ..core.policy import NeuralPolicy
from ..core.control_ode import ControlODE, ControlODEConfig
from ..core.system import TripleIntegrator
from ..utils.core_functions import TripleIntegratorCost, ConstrainedCost
from ..optimization.barriers import BarrierMethod, BarrierConfig

def objective_from_trajectory(trajectory: torch.Tensor, dt: float) -> torch.Tensor:
    state_term = trajectory[:, 0] ** 2
    velocity_penalty = 1e-3 * trajectory[:, 1] ** 2
    acceleration_penalty = 1e-3 * trajectory[:, 2] ** 2
    return dt * torch.sum(state_term + velocity_penalty + acceleration_penalty)

def optimize_unconstrained(cost_fn, params, steps: int = 400, lr: float = 5e-3) -> torch.Tensor:
    theta = torch.nn.Parameter(params.clone().detach())
    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = cost_fn(theta)
        loss.backward()
        optimizer.step()

    return theta.detach()

def run_triple_integrator():
    initial_state = torch.tensor([2.0, -2.0, -1.0])
    config = ControlODEConfig(tspan=(0.0, 1.34), dt=0.01)

    policy = NeuralPolicy(
        input_dim=3,
        hidden_dims=(12, 12),
        output_dim=1,
        activation="tanh",
        control_bounds=(-10.0, 10.0),
    )

    system = TripleIntegrator()
    ode = ControlODE(policy, system, initial_state, config)

    base_params = policy.get_parameter_vector()
    cost = TripleIntegratorCost(ode)

    unconstrained_params = optimize_unconstrained(cost, base_params)

    constraints = {
        "velocity": {"type": "state", "variable": 1, "bounds": (-30.0, 30.0)},
        "acceleration": {"type": "state", "variable": 2, "bounds": (-30.0, 30.0)},
    }

    constrained_cost = ConstrainedCost(
        ode,
        objective_fn=lambda traj: objective_from_trajectory(traj, config.dt),
        constraints=constraints,
        config={"alpha": 1.0, "delta": 10.0, "rho": 0.0},
    )

    barrier = BarrierMethod(BarrierConfig(max_iterations=25))
    optimizer_config = {"learning_rate": 5e-3, "max_steps": 200, "rho": 0.0}
    constrained_params, history = barrier.constrained_optimization(
        constrained_cost, unconstrained_params, optimizer_config
    )

    return {
        "ode": ode,
        "unconstrained_params": unconstrained_params,
        "constrained_params": constrained_params,
        "training_history": history,
    }

def plot_trajectory(ode: ControlODE, params: torch.Tensor, bounds=(-3.0, 3.0)) -> None:
    """Simulate the system with given params and plot states + control."""
    ode.policy.load_parameter_vector(params)
    trajectory = ode.solve(params).detach()
    times = ode.tsteps.detach()

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    labels = ["Position", "Velocity", "Acceleration"]

    for idx, label in enumerate(labels):
        axes[idx].plot(times.cpu().numpy(), trajectory[:, idx].cpu().numpy(), label=label)
        axes[idx].set_ylabel(label)
        axes[idx].grid(True)
        if label in ("Velocity", "Acceleration"):
            axes[idx].axhline(y=bounds[0], color="r", linestyle="--", alpha=0.6)
            axes[idx].axhline(y=bounds[1], color="r", linestyle="--", alpha=0.6)
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


if __name__ == "__main__":
    results = run_triple_integrator()
    print("Barrier iterations:", results["training_history"]["iterations"])
    plot_trajectory(results["ode"], results["constrained_params"])

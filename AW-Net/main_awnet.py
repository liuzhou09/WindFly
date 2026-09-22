"""Train Active Wind-Aware Network (AW-Net) with differentiable CUDA physics.

Notation follows the manuscript where possible:
    I_k                     onboard depth image
    x_dot, x_dot_ref         velocity and reference velocity [m/s]
    x_ddot_d                desired acceleration before compensation [m/s^2]
    d_hat            disturbance estimate 
    a_T_parallel_min/max    derived signed thrust acceleration projections [m/s^2]
    x_ddot_min/max_parallel target-direction acceleration bounds [m/s^2]
    x_ddot_lim_parallel     selected acceleration boundary [m/s^2]
    a_T_max                 maximum thrust acceleration magnitude [m/s^2]
    pi_theta, h_k           AW-Net policy and recurrent state
    N, r_k                  rollout length and obstacle displacement samples
    loss_bounds             dual-sided envelope penalty, final-paper Eq. (18)
    loss_f                  active-boundary tracking, final-paper Eq. (19)
    loss_w                  aerodynamic efficiency objective, final-paper Eq. (20)
    loss_mag                additional nominal magnitude penalty from main_cuda.py

The directional capability envelope uses the simulator or RA-Net disturbance
estimate d_hat.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import shlex


class ArgumentParser(argparse.ArgumentParser):
    """Accept the repository's whitespace-separated .args files."""

    def convert_arg_line_to_args(self, arg_line):
        """Split one argument-file line, respecting quotes and comments."""
        return shlex.split(arg_line, comments=True)


def parse_args(argv=None):
    """Parse training settings and reject unsupported numerical configurations."""
    parser = ArgumentParser(description=__doc__, fromfile_prefix_chars="@",
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", "--resume", type=Path,
                        help="Initialize policy weights; optimizer state is reset.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/awnet"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", "--batch_size", "--batch", type=int, default=64)
    parser.add_argument("--num-iters", "--num_iters", type=int, default=50000)
    parser.add_argument("--timesteps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-decay", "--grad_decay", type=float, default=0.4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--max-wind-speed", type=float, default=10.0,
                        help="Final wind curriculum level [m/s].")
    parser.add_argument("--wind-curriculum-iters", type=int, default=5000)
    parser.add_argument("--a-T-max", "--max_thrust", dest="a_T_max", type=float,
                        default=20.0,
                        help="Given thrust magnitude limit per unit mass, F_max / m [m/s^2].")
    parser.add_argument("--beta", "--barrier_beta", type=float, default=10.0,
                        help="Softplus sharpness for normalized violations.")

    
    weights = [
        ("velocity", "coef_v", 1.0, "Velocity tracking"),
        ("velocity-estimation", "coef_v_pred", 2.0, "Velocity estimation"),
        ("clearance", "coef_obj_avoidance", 1.5, "Obstacle clearance"),
        ("collision", "coef_collide", 2.0, "Collision avoidance"),
        ("acceleration", "coef_d_acc", 0.01, "Command regularization"),
        ("jerk", "coef_d_jerk", 0.001, "Command jerk regularization"),
        ("mag", "coef_thrust_barrier", 0.05, "Nominal magnitude penalty"),
        ("wind", "coef_wind_ride", 0.05, "Crosswind alignment"),
    ]
    for name, legacy_name, default, description in weights:
        parser.add_argument(f"--lambda-{name}", f"--{legacy_name}",
                            dest="lambda_" + name.replace("-", "_"),
                            type=float, default=default, help=description + " weight.")
    parser.add_argument("--lambda-bounds", "--lambda-env", "--coef_envelope_barrier",
                        dest="lambda_bounds", type=float, default=0.05,
                        help="Directional envelope weight; final-paper Eq. (18).")
    parser.add_argument("--lambda-f", "--lambda-boundary", "--coef_amax",
                        dest="lambda_f", type=float, default=0.7,
                        help="Active-boundary tracking weight; final-paper Eq. (19).")

    parser.add_argument("--speed-mtp", "--speed_mtp", type=float, default=1.0)
    parser.add_argument("--fov-x-half-tan", "--fov_x_half_tan", type=float, default=0.53)
    parser.add_argument("--cam-angle", "--cam_angle", type=int, default=10)
    for name in ("single", "gate", "ground_voxels", "scaffold",
                 "random_rotation", "yaw_drift", "no_odom"):
        flags = list(dict.fromkeys(("--" + name.replace("_", "-"), "--" + name)))
        parser.add_argument(*flags, dest=name, action="store_true")
    args = parser.parse_args(argv)

    for name in ("batch_size", "num_iters", "log_every", "checkpoint_every",
                 "wind_curriculum_iters", "lr", "a_T_max", "beta",
                 "speed_mtp", "fov_x_half_tan"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.timesteps <= 30:
        parser.error("timesteps must exceed the 30-step velocity averaging window")
    if not 0 < args.grad_decay <= 1:
        parser.error("grad_decay must lie in (0, 1]")
    if not args.single and args.batch_size % 8:
        parser.error("multi-agent batches must be divisible by 8 for Env grouping")
    for name, value in vars(args).items():
        if name.startswith("lambda_") or name == "max_wind_speed":
            if not math.isfinite(value) or value < 0:
                parser.error(f"{name} must be finite and nonnegative")
    return args


def acceleration_bounds(n_k, g, d_hat, a_T_max):
    """Return dynamic scalar bounds along n_k using the source capability model.

    All quantities are mass normalized. Gravity and disturbance are split into
    target-direction and transverse components. The signed directional thrust
    budget is derived after transverse compensation, not configured separately.
    a_T_max corresponds to a given scalar thrust magnitude limit F_max / m.
    """
    import torch
    from torch.nn import functional as F

    a_external = g + d_hat
    a_external_parallel = (a_external * n_k).sum(dim=-1, keepdim=True)
    a_external_perp = a_external - a_external_parallel * n_k
    a_external_perp_norm = a_external_perp.norm(p=2, dim=-1, keepdim=True)
    # Derived signed projection endpoints, not collective thrust magnitudes.
    a_T_parallel_max = torch.sqrt(
        F.relu(a_T_max**2 - a_external_perp_norm**2) + 1e-6)
    a_T_parallel_min = -a_T_parallel_max
    # Preserve the source numerical clipping; +/-1.5*a_T_max are not force limits.
    x_ddot_min_parallel = (a_external_parallel + a_T_parallel_min).clamp(
        -1.5 * a_T_max, 1.5 * a_T_max)
    x_ddot_max_parallel = (a_external_parallel + a_T_parallel_max).clamp(
        -1.5 * a_T_max, 1.5 * a_T_max)
    return x_ddot_min_parallel, x_ddot_max_parallel


def boundary_losses(x_ddot_d, n_k, g, x_ddot_min_parallel,
                    x_ddot_max_parallel, x_ddot_lim_parallel, a_T_max, beta):
    """Return loss_f (Eq. 19), loss_bounds (Eq. 18), and extra loss_mag.

    """
    from torch.nn import functional as F

    x_ddot_d_parallel = (x_ddot_d * n_k).sum(dim=-1, keepdim=True)
    # Eq. (19): active-boundary tracking, with the source Smooth-L1 objective.
    loss_f = F.smooth_l1_loss(
        x_ddot_d_parallel / a_T_max, x_ddot_lim_parallel.detach() / a_T_max)
    # Eq. (18): the two margins are scalars along the target direction.
    violation_upper = F.softplus(
        (x_ddot_d_parallel - x_ddot_max_parallel.detach()) / a_T_max, beta=beta)
    violation_lower = F.softplus(
        (x_ddot_min_parallel.detach() - x_ddot_d_parallel) / a_T_max, beta=beta)
    loss_bounds = (violation_upper.square() + violation_lower.square()).mean()
    a_T_nominal = x_ddot_d - g
    violation_magnitude = F.softplus(
        (a_T_nominal.norm(p=2, dim=-1, keepdim=True) - a_T_max) / a_T_max,
        beta=beta)
    loss_mag = violation_magnitude.square().mean()
    return loss_f, loss_bounds, loss_mag


def saturated_command(x_ddot_d, g, d_hat, a_T_max):
    """Compensate disturbance, limit thrust, and use Env's gravity convention."""
    import torch

    a_T_requested = x_ddot_d - g - d_hat
    a_T_norm = a_T_requested.norm(p=2, dim=-1, keepdim=True) + 1e-6
    scale = torch.clamp_max(a_T_norm, a_T_max) / a_T_norm
    return a_T_requested * scale + g


def wind_efficiency_loss(x_dot, x_dot_ref, wind_velocity, clearance_ahead):
    """Return the source crosswind implementation of the Eq. (20) objective.
    """
    import torch

    n_wind = x_dot_ref / (x_dot_ref.norm(p=2, dim=-1, keepdim=True) + 1e-6)
    wind_parallel = (wind_velocity * n_wind).sum(dim=-1, keepdim=True)
    wind_perp = wind_velocity - wind_parallel * n_wind
    velocity_parallel = (x_dot * n_wind).sum(dim=-1, keepdim=True)
    velocity_perp = x_dot - velocity_parallel * n_wind
    crosswind_alignment = (velocity_perp * wind_perp).sum(dim=-1)
    return -(crosswind_alignment[:, None] * torch.exp(-clearance_ahead)).mean()


def rollout_objective(args, env, pi_theta, iteration):
    """Roll out a batch; return the differentiable objective and diagnostics."""
    import torch
    from torch.nn import functional as F

    N = min(args.timesteps, 40 + (iteration // 100) * 5)
    wind_level = min(args.max_wind_speed,
                     args.max_wind_speed * iteration / args.wind_curriculum_iters)
    env.reset(wind_level=wind_level)
    pi_theta.reset()
    g = env.g_std
    device = g.device
    batch_size = args.batch_size
    history = defaultdict(list)
    h_k = None
    # Preserve the source command pipeline: two initial commands before outputs.
    command_history = [env.act, env.act]
    r_target = env.p_target - env.p

    if args.yaw_drift:
        yaw_increment = torch.randn(batch_size, device=device) * math.radians(5) / 15
        zeros, ones = torch.zeros_like(yaw_increment), torch.ones_like(yaw_increment)
        R_drift = torch.stack([
            yaw_increment.cos(), -yaw_increment.sin(), zeros,
            yaw_increment.sin(), yaw_increment.cos(), zeros,
            zeros, zeros, ones,
        ], dim=-1).reshape(batch_size, 3, 3)

    for k in range(N):
        delta_t = random.normalvariate(1 / 15, 0.1 / 15)
        I_k, _ = env.render(delta_t)
        r_k = env.find_vec_to_nearest_pt()
        history["r_obstacle"].append(r_k)
        if args.yaw_drift:
            r_target = (r_target[:, None] @ R_drift).squeeze(1)
        else:
            r_target = env.p_target - env.p.detach()
        env.run(command_history[k], delta_t, r_target)

        heading = env.R[:, :, 0].clone()
        heading[:, 2] = 0
        heading = F.normalize(heading, p=2, dim=-1)
        vertical = torch.zeros_like(heading)
        vertical[:, 2] = 1
        R_yaw = torch.stack(
            [heading, torch.cross(vertical, heading, dim=-1), vertical], dim=-1)
        r_target_norm = r_target.norm(p=2, dim=-1, keepdim=True)
        n_k = r_target / r_target_norm.clamp_min(1e-6)
        x_dot_ref = n_k * torch.minimum(r_target_norm, env.max_speed)
        x_dot = env.v

        if k % 5 == 0:
            # Disturbance estimate from simulation or RA-Net.
            d_hat = env.get_wind_disturbance(noise_std=0.05)
        d_hat_over_m_local = (d_hat[:, None] @ R_yaw).squeeze(1)
        d_hat_over_m_input = (d_hat_over_m_local / args.a_T_max).clamp(-2.0, 2.0)
        x_dot_local = (x_dot[:, None] @ R_yaw).squeeze(1)
        obs_k = [
            (x_dot_ref[:, None] @ R_yaw).squeeze(1),
            env.R[:, 2], env.margin[:, None],
        ]
        if not args.no_odom:
            obs_k.insert(0, x_dot_local)

        x_ddot_min_parallel, x_ddot_max_parallel = acceleration_bounds(
            n_k, g, d_hat, args.a_T_max)
        e_vel_parallel = x_dot_ref.norm(p=2, dim=-1, keepdim=True) - (
            x_dot * n_k).sum(dim=-1, keepdim=True)
        x_ddot_lim_parallel = torch.where(
            e_vel_parallel > 0, x_ddot_max_parallel, x_ddot_min_parallel)
        x_ddot_lim_input = (x_ddot_lim_parallel / args.a_T_max).clamp(-2.0, 2.0)
        I_input = 3 / I_k.clamp(0.3, 24) - 0.6 + torch.randn_like(I_k) * 0.02
        I_input = F.max_pool2d(I_input[:, None], 4, 4)
        policy_output, _, h_k = pi_theta(
            I_input, torch.cat(obs_k, dim=-1), d_hat_over_m_input, x_ddot_lim_input, h_k)
        x_ddot_head, x_dot_hat = (
            R_yaw @ policy_output.reshape(batch_size, 3, -1)).unbind(-1)
        x_ddot_d = (x_ddot_head - x_dot_hat - g) * env.thr_est_error[:, None] + g
        command_history.append(saturated_command(
            x_ddot_d, g, d_hat, args.a_T_max))

        for name, value in (
            ("x_dot", x_dot), ("x_dot_ref", x_dot_ref), ("x_dot_hat", x_dot_hat),
            ("x_ddot_d", x_ddot_d), ("n", n_k),
            ("x_ddot_min_parallel", x_ddot_min_parallel.detach()),
            ("x_ddot_max_parallel", x_ddot_max_parallel.detach()),
            ("wind_velocity", env.v_wind.detach()),
        ):
            history[name].append(value)

    trajectory = {name: torch.stack(values) for name, values in history.items()}
    x_dot = trajectory["x_dot"]
    x_dot_ref = trajectory["x_dot_ref"]
    commands = torch.stack(command_history)
    # Keep the original 30-step velocity averaging and reference alignment.
    x_dot_cumulative = x_dot.cumsum(dim=0)
    x_dot_average = (x_dot_cumulative[30:] - x_dot_cumulative[:-30]) / 30
    e_vel_norm = (x_dot_average - x_dot_ref[1:-29]).norm(p=2, dim=-1)
    loss_velocity = F.smooth_l1_loss(e_vel_norm, torch.zeros_like(e_vel_norm))
    loss_velocity_estimation = F.mse_loss(trajectory["x_dot_hat"], x_dot.detach())
    loss_acceleration = commands.square().sum(dim=-1).mean()
    loss_jerk = (commands.diff(dim=0) * 15).square().sum(dim=-1).mean()

    # Reproduce the source's loss-time direction and active-boundary selection.
    reference_speed = x_dot_ref.norm(p=2, dim=-1)
    n_ref = x_dot_ref / reference_speed[..., None].clamp_min(1e-6)
    forward_speed = (x_dot * n_ref).sum(dim=-1)
    x_ddot_lim_parallel = torch.where(
        (reference_speed - forward_speed)[..., None] > 0,
        trajectory["x_ddot_max_parallel"], trajectory["x_ddot_min_parallel"])
    loss_f, loss_bounds, loss_mag = boundary_losses(
        trajectory["x_ddot_d"], trajectory["n"], g,
        trajectory["x_ddot_min_parallel"], trajectory["x_ddot_max_parallel"],
        x_ddot_lim_parallel, args.a_T_max, args.beta)

    # Shape: [rollout time, look-ahead sample, batch]. Dimension 1 is not time.
    clearance = trajectory["r_obstacle"].norm(p=2, dim=-1) - env.margin
    with torch.no_grad():
        closing_speed = (-clearance.diff(dim=1) * 135).clamp_min(1)
    clearance_ahead = clearance[:, 1:]
    loss_clearance = (closing_speed * (1 - clearance_ahead).relu().square()).mean()
    loss_collision = (F.softplus(-32 * clearance_ahead) * closing_speed).mean()

    # Eq. (20): retain the original crosswind alignment and spatial weighting.
    loss_w = wind_efficiency_loss(
        x_dot, x_dot_ref, trajectory["wind_velocity"], clearance_ahead)

    loss_base = (
        args.lambda_velocity * loss_velocity
        + args.lambda_velocity_estimation * loss_velocity_estimation
        + args.lambda_clearance * loss_clearance
        + args.lambda_collision * loss_collision
        + args.lambda_acceleration * loss_acceleration
        + args.lambda_jerk * loss_jerk)

    loss_total = loss_base + (
        args.lambda_f * loss_f + args.lambda_bounds * loss_bounds
        + args.lambda_mag * loss_mag) + args.lambda_wind * loss_w
    losses = dict(total=loss_total, base=loss_base, bounds=loss_bounds, f=loss_f, w=loss_w,
                  velocity=loss_velocity, velocity_estimation=loss_velocity_estimation,
                  clearance=loss_clearance, collision=loss_collision,
                  acceleration=loss_acceleration, jerk=loss_jerk,
                  mag=loss_mag)
    with torch.no_grad():
        speed = x_dot.norm(p=2, dim=-1)
        collision_free = (clearance.flatten(0, 1) > 0).all(dim=0)
        metrics = {
            "collision_free_fraction": collision_free.float().mean(),
            "mean_speed": speed.mean(),
            "mean_peak_speed": speed.max(dim=0).values.mean(),
            "collision_free_speed": (collision_free * speed.mean(dim=0)).mean(),
            "horizon": N, "wind_level": wind_level,
        }
    return loss_total, losses, metrics


def train(args):
    """Initialize CUDA training, optimize the policy, and save logs and weights."""
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    from env_awnet import Env
    from model import Model

    if not torch.cuda.is_available():
        raise RuntimeError("AW-Net training requires a CUDA-enabled PyTorch installation.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {name: str(value) if isinstance(value, Path) else value
                  for name, value in vars(args).items()}
    (args.output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    env = Env(args.batch_size, 64, 48, args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan, single=args.single,
              gate=args.gate, ground_voxels=args.ground_voxels,
              scaffold=args.scaffold, speed_mtp=args.speed_mtp,
              random_rotation=args.random_rotation, cam_angle=args.cam_angle)
    pi_theta = Model(7 if args.no_odom else 10, 6).to(device)
    if args.weights:
        pi_theta.load_state_dict(torch.load(args.weights, map_location=device,
                                           weights_only=True))
    optimizer = AdamW(pi_theta.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.num_iters, args.lr * 0.01)
    pending_metrics = defaultdict(list)
    print(json.dumps(run_config, indent=2))
    with SummaryWriter(log_dir=str(args.output_dir)) as writer:
        progress = tqdm(range(args.num_iters), ncols=80)
        for iteration in progress:
            loss_total, losses, metrics = rollout_objective(args, env, pi_theta, iteration)
            if not torch.isfinite(loss_total).item():
                raise FloatingPointError(f"Non-finite training loss at iteration {iteration + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            progress.set_description(f"loss: {loss_total.detach().item():.3f}")
            for name, value in losses.items():
                pending_metrics[f"loss/{name}"].append(value.detach().item())
            for name, value in metrics.items():
                pending_metrics[f"rollout/{name}"].append(float(value))
            step = iteration + 1
            if step % args.log_every == 0 or step == args.num_iters:
                for name, values in pending_metrics.items():
                    writer.add_scalar(name, sum(values) / len(values), step)
                writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
                pending_metrics.clear()
            if step % args.checkpoint_every == 0 or step == args.num_iters:
                torch.save(pi_theta.state_dict(), args.output_dir / f"awnet_{step:06d}.pth")


def main():
    """Run the training entry point after parsing command-line arguments."""
    train(parse_args())


if __name__ == "__main__":
    main()

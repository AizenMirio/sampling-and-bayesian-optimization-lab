"""ULA and MALA samplers for neural energy models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .neural_energy_model import FEAT_DIM, default_device, load_energy_regressor, require_torch, seed_torch

try:
    import torch
except ModuleNotFoundError:
    torch = None


class UlaSampler:
    """Unadjusted Langevin Algorithm for differentiable energy models."""

    def __init__(self, model, step_size=1e-2, num_steps=100, noise_scale=1.0, device=None):
        require_torch()
        self.device = device or default_device()
        self.model = model.to(self.device)
        self.step_size = float(step_size)
        self.num_steps = int(num_steps)
        self.noise_scale = float(noise_scale)

    def sample(self, batch_size=64, dim=FEAT_DIM):
        x = torch.randn(batch_size, dim, device=self.device, requires_grad=True)
        start_time = time.time()
        step_scale = torch.sqrt(torch.tensor(self.step_size, device=self.device))
        for _ in range(self.num_steps):
            energy = self.model(x).sum()
            grad = torch.autograd.grad(energy, x, create_graph=False)[0]
            x = x - 0.5 * self.step_size * grad + self.noise_scale * step_scale * torch.randn_like(x)
            x = x.detach().requires_grad_(True)
        return x.detach(), time.time() - start_time


class MalaSampler:
    """Metropolis-Adjusted Langevin Algorithm for differentiable energy models."""

    def __init__(self, model, step_size=1e-2, num_steps=100, device=None):
        require_torch()
        self.device = device or default_device()
        self.model = model.to(self.device)
        self.step_size = float(step_size)
        self.num_steps = int(num_steps)

    def sample(self, batch_size=64, dim=FEAT_DIM):
        x = torch.randn(batch_size, dim, device=self.device)
        accepted = 0
        total = 0
        start_time = time.time()
        step_scale = torch.sqrt(torch.tensor(self.step_size, device=self.device))

        for _ in range(self.num_steps):
            x = x.detach().requires_grad_(True)
            energy = self.model(x).sum()
            grad = torch.autograd.grad(energy, x)[0]
            proposal = x - 0.5 * self.step_size * grad + step_scale * torch.randn_like(x)
            proposal = proposal.detach().requires_grad_(True)

            e_x = self.model(x).flatten()
            e_proposal = self.model(proposal).flatten()
            grad_proposal = torch.autograd.grad(e_proposal.sum(), proposal)[0]

            log_q_forward = -((proposal - x + 0.5 * self.step_size * grad) ** 2).sum(dim=1) / (
                2.0 * self.step_size
            )
            log_q_backward = -(
                (x - proposal + 0.5 * self.step_size * grad_proposal) ** 2
            ).sum(dim=1) / (2.0 * self.step_size)

            log_accept_ratio = -e_proposal + e_x + log_q_backward - log_q_forward
            accept_prob = torch.exp(torch.clamp(log_accept_ratio, max=0.0))
            accept_mask = torch.rand_like(accept_prob) < accept_prob
            accepted += int(accept_mask.sum().item())
            total += batch_size
            x = torch.where(accept_mask[:, None], proposal.detach(), x.detach())

        acceptance_rate = accepted / total if total else 0.0
        return x.detach(), time.time() - start_time, acceptance_rate


def _parse_args():
    parser = argparse.ArgumentParser(description="Sample from a neural energy model with ULA and MALA.")
    parser.add_argument("--weights", required=True, help="Path to model state dict (.pth).")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dim", type=int, default=FEAT_DIM)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/neural_energy")
    return parser.parse_args()


def main():
    args = _parse_args()
    require_torch()
    seed_torch(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_energy_regressor(args.weights, input_size=args.dim)

    ula_samples, ula_time = UlaSampler(model, args.step_size, args.steps).sample(args.batch_size, args.dim)
    mala_samples, mala_time, acceptance_rate = MalaSampler(model, args.step_size, args.steps).sample(
        args.batch_size,
        args.dim,
    )
    torch.save(
        {
            "ula": ula_samples.cpu(),
            "mala": mala_samples.cpu(),
            "ula_time": ula_time,
            "mala_time": mala_time,
            "mala_acceptance_rate": acceptance_rate,
        },
        output_dir / "neural_langevin_samples.pt",
    )
    print(
        f"ula_time={ula_time:.3f}s, mala_time={mala_time:.3f}s, "
        f"mala_acceptance_rate={acceptance_rate:.4f}"
    )


if __name__ == "__main__":
    main()


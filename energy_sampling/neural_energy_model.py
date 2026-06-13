"""Neural energy model utilities for the original high-dimensional experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # Keep the package importable without neural extras.
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


FEAT_DIM = 784


def require_torch():
    """Raise a clear error when optional neural dependencies are missing."""
    if torch is None:
        raise ModuleNotFoundError(
            "The neural energy reference requires PyTorch. Install optional "
            "dependencies with: python -m pip install -r requirements-neural.txt"
        )


def default_device():
    """Return the preferred torch device."""
    require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=42):
    """Seed torch for reproducible neural-energy evaluation."""
    require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EnergyDataset(Dataset):
    """Dataset wrapper for tensors stored as {'x': ..., 'energy': ...}."""

    def __init__(self, filepath):
        require_torch()
        data = torch.load(filepath, map_location="cpu")
        self.x = data["x"]
        self.energy = data["energy"].unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.energy[idx]


class EnergyRegressor(nn.Module if nn is not None else object):
    """Feed-forward regressor mapping a flattened 28x28 input to scalar energy."""

    def __init__(self, input_size=FEAT_DIM):
        require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 2),
            nn.ReLU(inplace=True),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        return self.net(x)


@dataclass(frozen=True)
class EvaluationResult:
    """Result from evaluating an energy regressor."""

    loss: float
    n_examples: int
    device: str


def load_energy_regressor(weights_path, input_size=FEAT_DIM, device=None):
    """Load an EnergyRegressor from a local state-dict file."""
    require_torch()
    if device is None:
        device = default_device()
    model = EnergyRegressor(input_size).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_energy_regressor(model, dataset_path, batch_size=1024, device=None):
    """Evaluate a loaded energy regressor with MSE loss."""
    require_torch()
    if device is None:
        device = default_device()
    dataset = EnergyDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    total_loss = 0.0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x_batch, energy_batch in loader:
            x_batch = x_batch.to(device)
            energy_batch = energy_batch.to(device)
            loss = criterion(model(x_batch), energy_batch)
            total_loss += float(loss.item()) * x_batch.size(0)
    return EvaluationResult(
        loss=total_loss / len(dataset),
        n_examples=len(dataset),
        device=str(device),
    )


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a neural energy regressor.")
    parser.add_argument("--weights", required=True, help="Path to model state dict (.pth).")
    parser.add_argument("--dataset", required=True, help="Path to tensor dataset (.pt).")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = _parse_args()
    weights_path = Path(args.weights)
    dataset_path = Path(args.dataset)
    seed_torch()
    model = load_energy_regressor(weights_path)
    result = evaluate_energy_regressor(model, dataset_path, args.batch_size)
    print(f"loss={result.loss:.6f}, n_examples={result.n_examples}, device={result.device}")


if __name__ == "__main__":
    main()


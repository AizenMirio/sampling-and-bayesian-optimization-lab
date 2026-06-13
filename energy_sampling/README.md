# Energy-Based Sampling

This folder contains ULA and MALA samplers for energy-based sampling.

## What It Implements

- `synthetic_langevin_demo.py`: a fully reproducible 2D double-well energy demo.
- `neural_energy_model.py`: a feed-forward energy regressor and evaluation
  helper for the original high-dimensional neural-energy experiment.
- `neural_langevin_samplers.py`: ULA and MALA samplers for differentiable neural
  energy models.

## Reproducible Demo

```bash
python -m energy_sampling.synthetic_langevin_demo --no-plot
```

This writes a compact sampler-diagnostics CSV under `outputs/synthetic_energy/`.

## Step-Size Sweep

```bash
python -m energy_sampling.compare_step_sizes --n-seeds 5
```

This writes:

- `docs/results/synthetic_energy_step_sweep_runs.csv`
- `docs/results/synthetic_energy_step_sweep_summary.csv`
- `docs/assets/synthetic_energy_step_sweep.svg`

## Optional Neural-Energy Reference

The neural-energy code requires PyTorch and local weights/data:

```bash
python -m pip install -r requirements-neural.txt
python -m energy_sampling.neural_energy_model --weights path/to/model.pth --dataset path/to/data.pt
python -m energy_sampling.neural_langevin_samplers --weights path/to/model.pth
```

## Result Summary

| Sampler | Burn-in Time | Main Tradeoff |
| --- | ---: | --- |
| ULA | 0.91s | Faster, approximate, no accept/reject correction |
| MALA | 2.28s | Slower, more principled target-distribution correction |

## Neural-Energy Reproducibility Note

The original model weights and test data are not included in this curation repo.
That part should be treated as an implementation reference unless shareable
weights/data are later added.

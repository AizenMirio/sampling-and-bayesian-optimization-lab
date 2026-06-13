import unittest

import numpy as np

from energy_sampling.synthetic_langevin_demo import (
    double_well_energy,
    double_well_grad,
    mala_sample,
    summarize_samples,
    ula_sample,
)
from energy_sampling import neural_energy_model


class EnergySamplingTests(unittest.TestCase):
    def test_energy_vectorization(self):
        values = double_well_energy(np.array([[-1.0, 0.0], [1.0, 0.0]]))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.allclose(values, 0.0))

    def test_gradient_shape(self):
        points = np.array([[-1.0, 0.5], [1.0, -0.5]])
        self.assertEqual(double_well_grad(points).shape, points.shape)

    def test_small_ula_run(self):
        samples = ula_sample(n_chains=32, n_steps=5, step_size=0.02, seed=0)
        self.assertEqual(samples.shape, (32, 2))
        summary = summarize_samples(samples)
        self.assertIn("mean_energy", summary)

    def test_small_mala_run(self):
        samples, acceptance_rate = mala_sample(n_chains=32, n_steps=5, step_size=0.02, seed=0)
        self.assertEqual(samples.shape, (32, 2))
        self.assertGreaterEqual(acceptance_rate, 0.0)
        self.assertLessEqual(acceptance_rate, 1.0)

    def test_neural_reference_imports_without_required_side_effects(self):
        if neural_energy_model.torch is None:
            with self.assertRaises(ModuleNotFoundError):
                neural_energy_model.require_torch()
        else:
            self.assertEqual(neural_energy_model.FEAT_DIM, 784)


if __name__ == "__main__":
    unittest.main()

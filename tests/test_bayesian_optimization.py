import unittest

import numpy as np

from bayesian_optimization.acquisition import expected_improvement, probability_of_improvement
from bayesian_optimization.benchmarks import branin_hoo, make_branin_grid
from bayesian_optimization.gp import GPParams, gaussian_process_predict
from bayesian_optimization.kernels import rbf_kernel
from bayesian_optimization.sequential_bo import BOConfig, run_sequential_bo


class BayesianOptimizationTests(unittest.TestCase):
    def test_branin_vectorization(self):
        values = branin_hoo(np.array([[-5.0, 0.0], [10.0, 15.0]]))
        self.assertEqual(values.shape, (2,))

    def test_gp_prediction_shapes(self):
        x_train = np.array([[-5.0, 0.0], [0.0, 5.0], [10.0, 15.0]])
        y_train = branin_hoo(x_train)
        _, _, x_grid, _ = make_branin_grid(grid_size=5)
        mu, sigma = gaussian_process_predict(
            x_train,
            y_train,
            x_grid,
            rbf_kernel,
            GPParams(length_scale=2.0, sigma_f=1.0, noise=1e-2),
        )
        self.assertEqual(mu.shape, (25,))
        self.assertEqual(sigma.shape, (25,))
        self.assertTrue(np.all(sigma >= 0))

    def test_acquisition_shapes(self):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.1, 0.2, 0.3])
        self.assertEqual(expected_improvement(mu, sigma, y_best=1.5).shape, (3,))
        self.assertEqual(probability_of_improvement(mu, sigma, y_best=1.5).shape, (3,))

    def test_small_sequential_bo_run(self):
        result = run_sequential_bo(
            BOConfig(n_initial=4, n_rounds=2, grid_size=8, seed=1, optimize_every=2)
        )
        self.assertEqual(len(result["records"]), 3)
        self.assertEqual(len(result["x_train"]), 6)


if __name__ == "__main__":
    unittest.main()

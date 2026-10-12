"""Independent numerical checks for the mathematical implementation."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from neural_network import NeuralNetwork, spiral_data, stratified_split, softmax, cross_entropy, gradient_check, spectral_analysis, separation_witness


class MathematicalChecks(unittest.TestCase):
    def test_full_gradient_relu_and_affine(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(2, 5))
        Y = np.eye(2)[[0, 1, 0, 1, 0]].T
        for activation in ("relu", "identity"):
            model = NeuralNetwork((2, 4, 3, 2), seed=17, activation=activation)
            check = gradient_check(model, X, Y)
            self.assertEqual(check["checked"], model.parameter_count)
            self.assertLess(check["max_absolute_error"], 1e-8)

    def test_softmax_and_loss_extreme_logits(self):
        Z = np.array([[1000., -1000.], [-1000., 1000.]])
        Y = np.eye(2)
        np.testing.assert_allclose(softmax(Z).sum(0), 1)
        self.assertEqual(cross_entropy(Z, Y), 0)
        self.assertEqual(cross_entropy(Z, Y[::-1]), 2000)
        np.testing.assert_allclose(softmax(Z), softmax(Z + 1e6))

    def test_split_disjoint_balanced_reproducible(self):
        X, Y = spiral_data()
        parts = stratified_split(Y)
        self.assertEqual([len(p) for p in parts], [240, 80, 80])
        self.assertEqual(len(set(np.concatenate(parts))), X.shape[1])
        for p in parts:
            self.assertEqual(Y[:, p].sum(1).tolist(), [len(p)/2]*2)
        np.testing.assert_array_equal(X, spiral_data()[0])

    def test_affine_stack_collapses_including_bias(self):
        model = NeuralNetwork(activation="identity")
        rng = np.random.default_rng(31)
        for key in model.params:
            if key.startswith("b"):
                model.params[key][:] = rng.normal(size=model.params[key].shape)
        effective_W, effective_b = np.eye(2), np.zeros((2, 1))
        for l in range(1, model.layers + 1):
            W, b = model.params[f"W{l}"], model.params[f"b{l}"]
            effective_W, effective_b = W @ effective_W, W @ effective_b + b
        X = rng.normal(size=(2, 10))
        np.testing.assert_allclose(model.forward(X)[1]["Z3"], effective_W @ X + effective_b, atol=1e-12)

    def test_svd_gram_and_nullspace(self):
        model = NeuralNetwork()
        self.assertEqual(model.parameter_count, 354)
        diagnostics = spectral_analysis(model)
        self.assertEqual(diagnostics[-1]["input_nullity"], 14)
        for d in diagnostics:
            self.assertLess(d["svd_reconstruction_relative_error"], 1e-12)
            self.assertLess(d["gram_identity_max_error"], 1e-12)

    def test_training_selects_the_recorded_validation_minimum(self):
        X, Y = spiral_data(20)
        tr, va, _ = stratified_split(Y)
        model = NeuralNetwork((2, 4, 2))
        model.fit(X[:, tr], Y[:, tr], X[:, va], Y[:, va], epochs=100, record_every=10)
        best = min(model.history, key=lambda h: h["validation_loss"])
        self.assertEqual(model.best_epoch, best["epoch"])
        self.assertAlmostEqual(model.evaluate(X[:, va], Y[:, va])["loss"], best["validation_loss"])

    def test_convex_hull_overlap_witness(self):
        X, Y = spiral_data()
        witness = separation_witness(X, Y)
        self.assertLess(witness["residual_norm"], 1e-12)
        a, b = witness["class0_indices"]
        c, d = witness["class1_indices"]
        self.assertEqual(Y[:, [a, b]].argmax(0).tolist(), [0, 0])
        self.assertEqual(Y[:, [c, d]].argmax(0).tolist(), [1, 1])
        t, u = witness["coefficients"]
        self.assertTrue(0 < t < 1 and 0 < u < 1)
        np.testing.assert_allclose((1-t)*X[:, a]+t*X[:, b], (1-u)*X[:, c]+u*X[:, d], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

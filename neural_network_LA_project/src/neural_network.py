"""Matrix-based classification: NumPy only, samples stored in columns.

The network combines affine maps, nonlinear activations, probability and
matrix calculus. Importing this module does not train or write any files.
"""
from __future__ import annotations

import numpy as np


def spiral_data(points_per_class=200, noise_std=0.0125, seed=42):
    """Two turns, Cartesian Gaussian noise, balanced labels and local RNG."""
    if points_per_class < 5 or noise_std < 0:
        raise ValueError("Need at least five points per class and nonnegative noise.")
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 4 * np.pi, points_per_class)
    radius = theta / (4 * np.pi)
    arms = [np.vstack((radius * np.cos(theta + c * np.pi),
                       radius * np.sin(theta + c * np.pi)))
            + rng.normal(0, noise_std, (2, points_per_class)) for c in range(2)]
    X = np.hstack(arms)
    labels = np.repeat(np.arange(2), points_per_class)
    Y = np.eye(2)[labels].T
    order = rng.permutation(X.shape[1])
    return X[:, order], Y[:, order]


def stratified_split(Y, seed=2026):
    """Fixed disjoint 60/20/20 split, stratified by class."""
    rng = np.random.default_rng(seed)
    groups = [[], [], []]
    for c in range(Y.shape[0]):
        indices = rng.permutation(np.flatnonzero(Y.argmax(axis=0) == c))
        a, b = int(0.6 * len(indices)), int(0.8 * len(indices))
        for group, part in zip(groups, (indices[:a], indices[a:b], indices[b:])):
            group.extend(part.tolist())
    return tuple(rng.permutation(group) for group in groups)


def softmax(logits):
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=0, keepdims=True)


def cross_entropy(logits, Y):
    """Mean cross-entropy using log-sum-exp, without clipping probabilities."""
    shifted = logits - logits.max(axis=0, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=0, keepdims=True))
    return float(-np.sum(Y * log_probs) / Y.shape[1])


class NeuralNetwork:
    def __init__(self, dimensions=(2, 16, 16, 2), seed=42, activation="relu"):
        if len(dimensions) < 2 or any(n < 1 for n in dimensions):
            raise ValueError("Layer dimensions must be positive.")
        if activation not in ("relu", "identity"):
            raise ValueError("Activation must be relu or identity.")
        self.dimensions = tuple(dimensions)
        self.activation = activation
        self.layers = len(dimensions) - 1
        rng = np.random.default_rng(seed)
        self.params = {}
        for l, (inputs, outputs) in enumerate(zip(dimensions[:-1], dimensions[1:]), 1):
            # He scaling for ReLU hidden maps, Xavier scaling for the output.
            scale = np.sqrt((2 if activation == "relu" and l < self.layers else 1) / inputs)
            self.params[f"W{l}"] = rng.normal(0, scale, (outputs, inputs))
            self.params[f"b{l}"] = np.zeros((outputs, 1))

    @property
    def parameter_count(self):
        return sum(p.size for p in self.params.values())

    def forward(self, X):
        if X.ndim != 2 or X.shape[0] != self.dimensions[0]:
            raise ValueError("X must have shape (input_dimension, samples).")
        cache = {"A0": X}
        A = X
        for l in range(1, self.layers + 1):
            Z = self.params[f"W{l}"] @ A + self.params[f"b{l}"]
            A = softmax(Z) if l == self.layers else (
                np.maximum(Z, 0) if self.activation == "relu" else Z)
            cache[f"Z{l}"], cache[f"A{l}"] = Z, A
        return A, cache

    def backward(self, Y, cache):
        """Delta is unaveraged. Divide parameter gradients by m exactly once."""
        m = Y.shape[1]
        delta = cache[f"A{self.layers}"] - Y
        grads = {}
        for l in range(self.layers, 0, -1):
            grads[f"W{l}"] = delta @ cache[f"A{l-1}"].T / m
            grads[f"b{l}"] = delta.sum(axis=1, keepdims=True) / m
            if l > 1:
                delta = self.params[f"W{l}"].T @ delta
                if self.activation == "relu":
                    delta *= cache[f"Z{l-1}"] > 0
        return grads

    def evaluate(self, X, Y):
        P, cache = self.forward(X)
        true, pred = Y.argmax(axis=0), P.argmax(axis=0)
        confusion = np.zeros((Y.shape[0], Y.shape[0]), dtype=int)
        np.add.at(confusion, (true, pred), 1)
        return {"loss": cross_entropy(cache[f"Z{self.layers}"], Y),
                "accuracy": float(np.mean(pred == true)),
                "correct": int(np.sum(pred == true)), "samples": X.shape[1],
                "confusion_matrix": confusion.tolist()}

    def fit(self, X, Y, X_val, Y_val, epochs=12000, learning_rate=0.1,
            record_every=50):
        """Full-batch GD. Select a checkpoint by validation loss, never test data.

        Epoch k means k completed updates, so metrics and saved parameters
        refer to the same state. Keep training for the full declared budget.
        """
        history, snapshots = [], []
        best_loss, best_epoch, best_params = float("inf"), 0, None
        for epoch in range(epochs + 1):
            P, cache = self.forward(X)
            if epoch % record_every == 0 or epoch == epochs:
                loss = cross_entropy(cache[f"Z{self.layers}"], Y)
                val = self.evaluate(X_val, Y_val)
                if not np.isfinite(loss + val["loss"]):
                    raise FloatingPointError("Non-finite loss. Reduce the learning rate.")
                history.append({"epoch": epoch, "train_loss": loss,
                                "train_accuracy": float(np.mean(P.argmax(0) == Y.argmax(0))),
                                "validation_loss": val["loss"],
                                "validation_accuracy": val["accuracy"]})
                if val["loss"] < best_loss:
                    best_loss, best_epoch = val["loss"], epoch
                    best_params = {k: v.copy() for k, v in self.params.items()}
                if epoch % 500 == 0 or epoch == epochs:
                    snapshots.append((epoch, {k: v.copy() for k, v in self.params.items()}))
            if epoch < epochs:
                for key, grad in self.backward(Y, cache).items():
                    self.params[key] -= learning_rate * grad
        self.params = best_params
        self.history, self.snapshots, self.best_epoch = history, snapshots, best_epoch
        return self


def spectral_analysis(model):
    """Diagnostics of individual affine weight maps, before ReLU."""
    result = []
    for l in range(1, model.layers + 1):
        W = model.params[f"W{l}"]
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        gram = np.linalg.eigvalsh(W.T @ W)
        expected = np.sort(np.r_[s * s, np.zeros(W.shape[1] - len(s))])
        result.append({"layer": l, "shape": list(W.shape),
                       "rank": int(np.linalg.matrix_rank(W)),
                       "singular_values": s.tolist(),
                       "condition_nonzero_spectrum": float(s[0] / s[-1]),
                       "input_nullity": int(W.shape[1] - np.linalg.matrix_rank(W)),
                       "top4_energy_fraction": float(np.sum(s[:4] ** 2) / np.sum(s ** 2)),
                       "gram_eigenvalues": gram.tolist(),
                       "gram_identity_max_error": float(np.max(np.abs(gram - expected))),
                       "svd_reconstruction_relative_error": float(
                           np.linalg.norm(W - (U * s) @ Vt) / np.linalg.norm(W))})
    return result


def gradient_check(model, X, Y, epsilon=1e-6):
    """Check every scalar using centered differences away from ReLU kinks."""
    _, cache = model.forward(X)
    analytic = model.backward(Y, cache)
    errors, absolute, skipped = [], [], 0
    def same_region(a, b):
        return model.activation != "relu" or all(
            np.array_equal(a[f"Z{l}"] > 0, b[f"Z{l}"] > 0)
            for l in range(1, model.layers))
    for key, values in model.params.items():
        for index in np.ndindex(values.shape):
            original = values[index]
            try:
                values[index] = original + epsilon
                _, plus = model.forward(X)
                values[index] = original - epsilon
                _, minus = model.forward(X)
            finally:
                values[index] = original
            if not same_region(plus, minus):
                skipped += 1
                continue
            numerical = (cross_entropy(plus[f"Z{model.layers}"], Y)
                         - cross_entropy(minus[f"Z{model.layers}"], Y)) / (2 * epsilon)
            a = analytic[key][index]
            absolute.append(abs(a - numerical))
            errors.append(abs(a - numerical) / max(1e-8, abs(a) + abs(numerical)))
    return {"checked": len(errors), "skipped_kinks": skipped,
            "max_absolute_error": float(max(absolute, default=0)),
            "max_relative_error": float(max(errors, default=0))}


def separation_witness(X, Y):
    """Find intersecting same-class segments: a convex-hull overlap witness.

    If two convex combinations from different classes coincide, no affine
    score can be strictly positive on one whole class and negative on the
    other. This proves impossibility of 100%, without imposing a 50% cap.
    """
    labels = Y.argmax(0)
    groups = [np.flatnonzero(labels == c) for c in (0, 1)]
    # Opposite extrema give long within-class segments and a small search.
    pairs = []
    for group in groups:
        candidates = np.unique([group[np.argmin(X[d, group])] for d in (0, 1)]
                               + [group[np.argmax(X[d, group])] for d in (0, 1)])
        pairs.append([(a, b) for i, a in enumerate(candidates) for b in candidates[i+1:]])
    for a, b in pairs[0]:
        for c, d in pairs[1]:
            M = np.column_stack((X[:, b] - X[:, a], -(X[:, d] - X[:, c])))
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            t, u = np.linalg.solve(M, X[:, c] - X[:, a])
            if 0 < t < 1 and 0 < u < 1:
                point0 = (1-t) * X[:, a] + t * X[:, b]
                point1 = (1-u) * X[:, c] + u * X[:, d]
                return {"class0_indices": [int(a), int(b)], "class1_indices": [int(c), int(d)],
                        "coefficients": [float(t), float(u)], "intersection": point0.tolist(),
                        "residual_norm": float(np.linalg.norm(point0 - point1))}
    raise ValueError("This search found no witness. It does not prove separability.")

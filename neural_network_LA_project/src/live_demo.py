"""Show the classifier learning, then save its held-out decision boundary."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from neural_network import NeuralNetwork, cross_entropy, separation_witness, spiral_data, stratified_split

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
DEMO_DIR = RESULTS / "live_demo"


def train_live(epochs=12_000, interval=0.45):
    """Fit the declared deep network, announcing validation progress as it learns."""
    X, Y = spiral_data(points_per_class=200, noise_std=0.0125, seed=42)
    train_idx, validation_idx, test_idx = stratified_split(Y, seed=2026)
    model = NeuralNetwork((2, 16, 16, 2), seed=42)
    X_train, Y_train = X[:, train_idx], Y[:, train_idx]
    X_val, Y_val = X[:, validation_idx], Y[:, validation_idx]
    best_loss, best_epoch, best_params = float("inf"), 0, None
    history = []
    interactive = sys.stdout.isatty()

    print("Two interleaved spirals | 400 points | matrix network 2-16-16-2")
    print("Train 240 | validate 80 | test 80 | NumPy only | learning rate 0.1")
    witness = separation_witness(X, Y)
    print(f"A straight separator cannot classify every point (hull overlap residual "
          f"{witness['residual_norm']:.1e}).")
    print("Each update uses Z = W @ A + b, ReLU, softmax and matrix transposes.\n")

    for epoch in range(epochs + 1):
        _, cache = model.forward(X_train)
        if epoch % 50 == 0 or epoch == epochs:
            train_loss = cross_entropy(cache[f"Z{model.layers}"], Y_train)
            val = model.evaluate(X_val, Y_val)
            if not np.isfinite(train_loss + val["loss"]):
                raise FloatingPointError("Training became non-finite; try a smaller learning rate.")
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "validation_loss": val["loss"],
                            "validation_accuracy": val["accuracy"]})
            if val["loss"] < best_loss:
                best_loss, best_epoch = val["loss"], epoch
                best_params = {key: value.copy() for key, value in model.params.items()}
            if epoch % 500 == 0 or epoch == epochs:
                state = "initialization" if epoch == 0 else f"{epoch:>5,} updates"
                print(f"{state:>18} | train loss {train_loss:.4f} "
                      f"| validation loss {val['loss']:.4f} "
                      f"| validation accuracy {val['accuracy']:.1%} "
                      f"| best checkpoint {best_epoch:>5,}", flush=True)
                if interactive and interval:
                    time.sleep(interval)
        if epoch < epochs:
            gradients = model.backward(Y_train, cache)
            for key, gradient in gradients.items():
                model.params[key] -= 0.1 * gradient

    model.params = best_params
    test = model.evaluate(X[:, test_idx], Y[:, test_idx])
    train = model.evaluate(X_train, Y_train)
    validation = model.evaluate(X_val, Y_val)
    print("\nValidation-selected network")
    print(f"Checkpoint: {best_epoch:,} updates | train {train['accuracy']:.1%} "
          f"| validation {validation['accuracy']:.1%} "
          f"| test {test['correct']}/{test['samples']} ({test['accuracy']:.1%})")
    print("Confusion matrix (rows=true class, columns=predicted class):")
    print(np.array(test["confusion_matrix"]))
    return model, X, Y, test_idx, history, {
        "epochs": epochs, "learning_rate": 0.1, "initialization_seed": 42,
        "split_seed": 2026, "selected_epoch": best_epoch,
        "train": train, "validation": validation, "test": test,
    }


def save_boundary(model, X, Y, test_idx, history, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    labels = Y.argmax(axis=0)
    class_colors = ["#3157b7", "#d05b50"]
    grid_x, grid_y = np.meshgrid(np.linspace(-1.15, 1.15, 220),
                                 np.linspace(-1.15, 1.15, 220))
    grid = np.vstack((grid_x.ravel(), grid_y.ravel()))
    probability = model.forward(grid)[0][1].reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    contour = ax.contourf(grid_x, grid_y, probability, levels=np.linspace(0, 1, 17),
                          cmap="coolwarm", alpha=0.62, vmin=0, vmax=1)
    ax.contour(grid_x, grid_y, probability, levels=[0.5], colors="#183c2d", linewidths=1.6)
    ax.scatter(X[0, test_idx], X[1, test_idx], c=labels[test_idx],
               cmap=ListedColormap(class_colors), vmin=0, vmax=1,
               s=35, edgecolors="white", linewidths=0.6)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", xlim=(-1.15, 1.15), ylim=(-1.15, 1.15),
           aspect="equal", title=(f"Two spirals | test accuracy {summary['test']['accuracy']:.1%}\n"
                                    f"Checkpoint selected at {summary['selected_epoch']:,} updates"))
    fig.colorbar(contour, ax=ax, label="$P(\\mathrm{class 1} \\mid x)$")
    image_path = DEMO_DIR / "decision_boundary.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    (DEMO_DIR / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (DEMO_DIR / "learning_history.csv").open("w") as stream:
        stream.write("epoch,train_loss,validation_loss,validation_accuracy\n")
        for row in history:
            stream.write(f"{row['epoch']},{row['train_loss']:.12g},"
                         f"{row['validation_loss']:.12g},{row['validation_accuracy']:.12g}\n")
    print(f"\nDecision boundary saved to: {image_path}")
    print(f"Metrics and learning history saved to: {DEMO_DIR}")
    return image_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12_000,
                        help="gradient updates (default: 12000; rounded down to a 50-update checkpoint)")
    parser.add_argument("--interval", type=float, default=0.45,
                        help="seconds between progress lines in an interactive terminal (default: 0.45)")
    parser.add_argument("--open", action="store_true",
                        help="open the final decision-boundary image in your default image viewer")
    args = parser.parse_args()
    if args.epochs < 50 or args.interval < 0:
        parser.error("--epochs must be at least 50 and --interval cannot be negative")
    epochs = args.epochs // 50 * 50
    if epochs != args.epochs:
        print(f"Using {epochs:,} updates to keep checkpoint selection on the 50-update schedule.")
    model, X, Y, test_idx, history, summary = train_live(epochs, args.interval)
    image_path = save_boundary(model, X, Y, test_idx, history, summary)
    if args.open:
        import webbrowser
        if not webbrowser.open(image_path.as_uri()):
            print("Could not open a viewer automatically. Open the image path above.")


if __name__ == "__main__":
    main()

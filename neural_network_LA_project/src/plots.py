"""Scientific figures, with held-out data and projection limits labeled."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

COLORS = ["#3157b7", "#d05b50"]
CMAP = ListedColormap(COLORS)


def make_figures(models, X, Y, train, val, test, destination, make_animation=True):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    def save(fig, name):
        fig.savefig(destination / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    def points(ax, indices, size=18):
        ax.scatter(X[0, indices], X[1, indices], c=Y[:, indices].argmax(0), cmap=CMAP,
                   vmin=0, vmax=1, s=size, edgecolors="white", linewidths=0.3)
        ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal", xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, indices, title in zip(axes, (train, val, test), ("Training (240)", "Validation (80)", "Test (80)")):
        points(ax, indices)
        ax.set_title(title)
    fig.suptitle("Two turns per spiral, Cartesian noise standard deviation 0.0125")
    save(fig, "dataset_split")
    gx, gy = np.meshgrid(np.linspace(-1.15, 1.15, 180), np.linspace(-1.15, 1.15, 180))
    grid = np.vstack((gx.ravel(), gy.ravel()))
    def boundary(ax, model):
        P = model.forward(grid)[0][1].reshape(gx.shape)
        ax.contourf(gx, gy, P, levels=np.linspace(0, 1, 15), cmap="coolwarm", alpha=.55, vmin=0, vmax=1)
        if P.min() < .5 < P.max():
            ax.contour(gx, gy, P, levels=[.5], colors="#24372d", linewidths=1)
        points(ax, test, 24)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (name, model) in zip(axes, models.items()):
        boundary(ax, model)
        score = model.evaluate(X[:, test], Y[:, test])["accuracy"]
        ax.set_title(f"{name.capitalize()}: test {score:.1%}")
    fig.suptitle("Selected checkpoints (seed 42). Dots are test data, contour is P(class 1) = 0.5")
    save(fig, "model_comparison")
    deep = models["deep"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epochs = [h["epoch"] for h in deep.history]
    for label, prefix, color in (("Training", "train", COLORS[0]), ("Validation", "validation", COLORS[1])):
        axes[0].plot(epochs, [h[prefix + "_loss"] for h in deep.history], label=label, color=color)
        axes[1].plot(epochs, [h[prefix + "_accuracy"] for h in deep.history], label=label, color=color)
    for ax in axes:
        ax.axvline(deep.best_epoch, color="#304e3c", linestyle="--", label=f"Selected epoch {deep.best_epoch}")
        ax.set_xlabel("Completed gradient updates")
        ax.legend(fontsize=9)
    axes[0].set(ylabel="Mean cross-entropy", title="Loss need not decrease at every update")
    axes[1].set(ylabel="Accuracy", ylim=(0, 1.02), title="Test set is absent from checkpoint selection")
    save(fig, "learning_curves")
    _, cache_train = deep.forward(X[:, train])
    _, cache_test = deep.forward(X[:, test])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    pca = []
    for ax, l in zip(axes[:2], (1, 2)):
        H = cache_train[f"A{l}"]
        mean = H.mean(axis=1, keepdims=True)
        U, s, _ = np.linalg.svd(H - mean, full_matrices=False)
        projection = U[:, :2].T @ (cache_test[f"A{l}"] - mean)
        retained = float(np.sum(s[:2] ** 2) / np.sum(s ** 2))
        pca.append({"layer": l, "training_variance_retained": retained})
        ax.scatter(*projection, c=Y[:, test].argmax(0), cmap=CMAP, vmin=0, vmax=1, s=20)
        ax.set(title=f"Hidden {l}: PCA, {retained:.1%} variance", xlabel="PC 1", ylabel="PC 2")
    logits = cache_test[f"Z{deep.layers}"]
    axes[2].scatter(*logits, c=Y[:, test].argmax(0), cmap=CMAP, vmin=0, vmax=1, s=20)
    lo, hi = logits.min(), logits.max()
    axes[2].plot([lo, hi], [lo, hi], "k--", label="$z_0=z_1$")
    axes[2].set(title="Test logits and exact readout boundary", xlabel="$z_0$", ylabel="$z_1$")
    axes[2].legend()
    fig.suptitle("Test representations. PCA bases fitted on training activations only")
    save(fig, "hidden_representations")
    import json
    (destination.parent / "pca.json").write_text(json.dumps(pca, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for l, ax in enumerate(axes, 1):
        W = deep.params[f"W{l}"]
        limit = np.abs(W).max()
        image = ax.imshow(W, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set(title=f"$W^{l}$: {W.shape}", xlabel="Input coordinate", ylabel="Output coordinate")
        fig.colorbar(image, ax=ax, shrink=.75)
    fig.suptitle("Learned weights. Each panel uses its own symmetric color scale")
    save(fig, "weight_heatmaps")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for l, ax in enumerate(axes, 1):
        s = np.linalg.svd(deep.params[f"W{l}"], compute_uv=False)
        ax.bar(np.arange(1, len(s)+1), s, color="#304e3c")
        ax.set(title=f"Layer {l}: rank {np.linalg.matrix_rank(deep.params[f'W{l}'])}",
               xlabel="Singular value index", ylabel="Singular value")
    fig.suptitle("Spectra of weight maps. ReLU and activation covariance require separate analysis")
    save(fig, "singular_values")
    fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    cm = np.array(deep.evaluate(X[:, test], Y[:, test])["confusion_matrix"])
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=22,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted class", ylabel="True class",
           title="Deep network test confusion matrix (seed 42)")
    save(fig, "confusion_matrix")
    if make_animation:
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        from neural_network import NeuralNetwork
        display_model = NeuralNetwork(deep.dimensions, seed=0)
        def frame(i):
            epoch, params = deep.snapshots[i]
            display_model.params = params
            ax.clear()
            boundary(ax, display_model)
            ax.set_title(f"Deep network: {epoch} completed updates\nDots are held-out test points")
        animation = FuncAnimation(fig, frame, frames=len(deep.snapshots), interval=250)
        animation.save(destination / "training_boundary.gif", writer=PillowWriter(fps=4))
        plt.close(fig)

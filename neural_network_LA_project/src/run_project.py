"""Run the declared experiment and regenerate evidence from actual arrays."""
from pathlib import Path
import argparse
import csv
import json
import platform
import numpy as np
from neural_network import NeuralNetwork, spiral_data, stratified_split, spectral_analysis, gradient_check, separation_witness

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"points_per_class": 200, "noise_std": 0.0125, "data_seed": 42,
          "split_seed": 2026, "initialization_seeds": [42, 43, 44],
          "epochs": 12000, "learning_rate": 0.1, "record_every": 50,
          "checkpoint_selection": "minimum validation cross-entropy at recorded epochs"}
ARCHITECTURES = {"linear": (2, 2), "shallow": (2, 16, 2), "deep": (2, 16, 16, 2)}


def run(output=ROOT / "results", make_animation=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    X, Y = spiral_data(CONFIG["points_per_class"], CONFIG["noise_std"], CONFIG["data_seed"])
    train, val, test = stratified_split(Y, CONFIG["split_seed"])
    np.savez_compressed(output / "dataset.npz", X=X, Y=Y, train=train, validation=val, test=test)
    models, runs = {}, []
    for name, dimensions in ARCHITECTURES.items():
        for seed in CONFIG["initialization_seeds"]:
            model = NeuralNetwork(dimensions, seed=seed)
            model.fit(X[:, train], Y[:, train], X[:, val], Y[:, val],
                      epochs=CONFIG["epochs"], learning_rate=CONFIG["learning_rate"],
                      record_every=CONFIG["record_every"])
            row = {"model": name, "seed": seed, "dimensions": list(dimensions),
                   "parameters": model.parameter_count, "selected_epoch": model.best_epoch,
                   "train": model.evaluate(X[:, train], Y[:, train]),
                   "validation": model.evaluate(X[:, val], Y[:, val]),
                   "test": model.evaluate(X[:, test], Y[:, test])}
            runs.append(row)
            with (output / f"history_{name}_{seed}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(model.history[0]))
                writer.writeheader()
                writer.writerows(model.history)
            np.savez_compressed(output / f"weights_{name}_{seed}.npz", **model.params)
            if seed == CONFIG["initialization_seeds"][0]:
                models[name] = model
            print(f"{name:7} seed {seed}: train={row['train']['accuracy']:.3%}, "
                  f"val={row['validation']['accuracy']:.3%}, test={row['test']['accuracy']:.3%}, "
                  f"epoch={model.best_epoch}", flush=True)
    check_model = NeuralNetwork((2, 4, 3, 2), seed=17)
    check = gradient_check(check_model, X[:, train[:5]], Y[:, train[:5]])
    deep = models["deep"]
    U, s, Vt = np.linalg.svd(deep.params["W2"], full_matrices=False)
    compression = []
    for k in (2, 4, 8, 12, 16):
        compressed = NeuralNetwork(deep.dimensions)
        compressed.params = {name: value.copy() for name, value in deep.params.items()}
        compressed.params["W2"] = (U[:, :k] * s[:k]) @ Vt[:k]
        error = float(np.linalg.norm(deep.params["W2"] - compressed.params["W2"], "fro"))
        compression.append({"rank": k, "weight_energy_retained": float(np.sum(s[:k]**2) / np.sum(s**2)),
                            "frobenius_error": error, "singular_tail_error": float(np.linalg.norm(s[k:])),
                            "factor_entries": 32*k, "original_entries": 256,
                            "validation": compressed.evaluate(X[:, val], Y[:, val]),
                            "test": compressed.evaluate(X[:, test], Y[:, test])})
    summary = {"config": CONFIG, "environment": {"python": platform.python_version(),
               "numpy": np.__version__}, "split_sizes": {"train": len(train), "validation": len(val),
               "test": len(test)}, "runs": runs, "gradient_check": check,
               "spectral_analysis": spectral_analysis(deep),
               "linear_nonseparability_witness": separation_witness(X, Y),
               "rank_truncation": compression,
               "seed_summary": {name: {"mean_test_accuracy": float(np.mean([
                   r["test"]["accuracy"] for r in runs if r["model"] == name])),
                   "std_test_accuracy": float(np.std([
                   r["test"]["accuracy"] for r in runs if r["model"] == name], ddof=1))}
                   for name in ARCHITECTURES}}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    from plots import make_figures
    make_figures(models, X, Y, train, val, test, output / "figures", make_animation)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    run(args.output, not args.no_animation)

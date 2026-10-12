# Neural networks through matrix operations

**UE25MA242A: Mathematical Foundation for AI & Data Science**

Research question: How do ReLU activations extend affine classifiers, and what do PCA and SVD tell us about the learned transformations?

## Live demonstration

From the Maths workspace, run:

```bash
./neural_network_LA_project/run_live_demo.sh
```

The script trains the deep network in the terminal, prints its training and validation progress, reports the final held-out test score, then saves the decision boundary under `results/live_demo/`. Use `--open` to open that image automatically. Use `--interval 0` to remove the brief pause between progress lines. Press Ctrl+C to stop training.

This is an inspired extension of the supplied Projects 10 (transformations), 11 (PCA) and 14 (SVD compression). Confirm that your instructor accepts extensions to the listed topics. Add your team names before submission.

## Submission files

- [Project report](docs/PROJECT_REPORT.pdf): nine pages of methods, derivations, measured results, limitations and references.
- [Editable PowerPoint](presentation/Neural_Network_Linear_Algebra.pptx): sixteen slides with speaker notes, editable tables and a native loss chart.
- [Browser presentation](presentation/presentation.html): opens locally, works offline, supports arrow keys and printing.
- [Notebook](notebooks/neural_network_project.ipynb): explanations and inspection of the shared implementation and saved results.
- [Speaker notes](docs/SPEAKER_NOTES.md) and [viva guide](docs/VIVA_GUIDE.md).

## Reproduce the experiment

Python 3.12 or later is recommended. The checked local environment uses Python 3.14.7, NumPy 2.5.3 and Matplotlib 3.11.2. Training requires no ML frameworks.

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/run_project.py
python -m unittest discover -s tests -v
```

The existing `.venv` already contains the experiment dependencies. You can also run `.venv/bin/python src/run_project.py` directly. Use `--no-animation` to skip GIF generation and `--output /your/path` to write results elsewhere. Matplotlib uses a noninteractive backend, so the script runs without a display.

For the notebook, install JupyterLab into that environment and launch it from here:

```bash
python -m pip install jupyterlab
python -m jupyter lab notebooks/neural_network_project.ipynb
```

The notebook loads saved evidence by default. Set `RETRAIN=True` to regenerate all nine runs. It imports the same source as the command-line script.

## Declared experiment

- 400 synthetic points, 200 per class, two turns per spiral.
- Independent Cartesian Gaussian noise with standard deviation 0.0125. Data seed 42.
- Stratified, disjoint 240/80/80 training/validation/test split, seed 2026.
- Models: linear `2-2` (6 parameters), shallow `2-16-2` (82), deep `2-16-16-2` (354).
- Full-batch gradient descent, 12,000 updates, learning rate 0.1, recorded every 50 updates.
- Checkpoint selected by minimum recorded validation cross-entropy. Test data do not influence selection.
- Initialization seeds 42, 43 and 44. Seed 42 is the declared primary run.

| Model, seed 42 | Train | Validation | Test | Selected epoch |
| --- | --- | --- | --- | --- |
| Linear | 50.00% | 45.00% | 58.75% | 4500 |
| Shallow | 57.50% | 51.25% | 58.75% | 50 |
| Deep | 97.08% | 93.75% | 90.00% | 11950 |

Deep test accuracy across the three initialization seeds is 90.00%, 92.50% and 78.75% (mean 87.08%, sample SD 7.32 percentage points). This variation matters. One small synthetic split does not establish real-world performance or prove that all shallow networks fail.

`results/metrics.json` is the authoritative numerical record. Small differences across NumPy/BLAS versions can occur in long nonlinear optimization runs. Each history row describes exactly its recorded number of completed updates. Reported scores use the restored selected checkpoint.

## Mathematics and checks

With samples in columns, `Z = W @ A + b` broadcasts the bias across the batch. Hidden layers use ReLU and the output uses stable softmax. Loss uses log-sum-exp from logits. Backpropagation uses transposes, ReLU masks and sums of outer products, averaging gradients exactly once.

Seven numerical checks cover scalar derivatives in small ReLU and affine networks, extreme-logit loss, balanced splitting, affine collapse with biases, SVD/Gram identities, an intersecting-segment nonseparability witness and checkpoint restoration. The saved experiment's gradient check reports a maximum absolute error below `1e-10`.

PCA bases fit **training activations**, then project test activations. Weight SVD is a separate analysis. Rectangular output matrices have structural nullspaces even at full row rank. A condition ratio does not prove stable gradients or removal of noise.

The rank truncation experiment replaces only `W2` by rank 2/4/8/12/16 SVD approximations. It compares approximation energy with classification accuracy and verifies the discarded-singular-value error identity. A factorization would use `32*k` entries versus 256 dense entries, but the experiment materializes approximations rather than storing compressed factors.

## Files and regeneration

`src/neural_network.py` contains the implementation and numerical diagnostics. Importing it has no filesystem or training side effects. `src/run_project.py` runs the declared experiment. `src/plots.py` creates figures. `tests/test_math.py` checks mathematical identities independently.

`results/` contains the dataset and split indices, nine selected weight files, nine CSV histories, numerical metrics and PCA variance ratios. `results/figures/` contains seven PNGs and one animation. The full training GIF ends at the final update budget, while reported results use the validation-selected checkpoint.

To rebuild the PDF, report Markdown, notebook, HTML presentation and speaker notes from regenerated results, install ReportLab and run:

```bash
python -m pip install reportlab
python src/build_writeup.py
```

The PPTX builder `src/build_slides.mjs` uses the Codex bundled `@oai/artifact-tool` and presentation validators. Its three runtime path variables are documented at the top of the script. It consumes `presentation/slides.json` and the measured loss history. Finalization requires a new output filename if a deck already exists. The core experiment runs independently of these artifact tools.

## References

Original files remain in [the workspace reference folder](../references/). The supplied adapted lab manual retains Anna Zemlyanova's CC BY-NC 4.0 attribution. It contains mixed MATLAB/Python fragments, so we use verified NumPy operations.

- Goodfellow, Bengio and Courville, [Deep Learning, Chapter 6](https://www.deeplearningbook.org/contents/mlp.html): feedforward networks and backpropagation.
- Stanford, [CS231n gradient checking](https://cs231n.github.io/neural-networks-3/): centered finite differences and nondifferentiable points.
- NumPy, [SVD documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html): singular values and returned factor conventions.

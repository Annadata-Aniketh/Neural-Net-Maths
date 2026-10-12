# Presentation speaker notes

The deck has 16 slides. A 10-12 minute talk can spend 30-45 seconds on most slides and about a minute on gradients and results. Use your own words.

## Slide 1: Neural networks through matrix operations

Introduce the research question: how do nonlinear activations extend affine classifiers, and how can matrix decompositions describe the learned maps? Our network uses NumPy for its calculations. It also relies on calculus, probability and optimization.

## Slide 2: Connection to the supplied projects

The sheet lists 14 projects, without a neural-network problem. Present this as an inspired extension and confirm that extensions are acceptable. The confirmed course code is UE25MA242A. Sources: supplied problem sheet and adapted Anna Zemlyanova manual, Projects 10, 11 and 14.

## Slide 3: Two spirals and a held-out test set

Data seed 42, split seed 2026. Every sample is a column of X. Test data do not participate in training or checkpoint selection. Coordinates already lie approximately within the unit disk.

## Slide 4: Why a line cannot classify every point

Use the saved convex hull witness in metrics.json. An affine score preserves convex combinations, so the intersection could not have opposite signs. This proves impossibility of perfect classification on these points. It does not limit affine accuracy to 50%. The measured linear test score can exceed 50%.

## Slide 5: Architecture and parameter count

The architecture is 2-16-16-2 with three trainable affine maps. The original draft added the totals incorrectly. Hidden weights use He variance scaling and output weights use Xavier scaling. Initialization is a heuristic rather than a guarantee of spectral stability.

## Slide 6: Vectorized forward propagation

For training, m=240. Z1 and Z2 have 16 rows, and Z3 has two rows. Bias broadcasting implements b times a row of ones. Without hidden nonlinearities, affine maps collapse to an effective W and b. Batch matrix computation eliminates loops over individual samples, but computational cost still grows with sample count and matrix size.

## Slide 7: Cross-entropy and matrix gradients

D is unaveraged, so divide by m exactly once in parameter gradients. The objective is mean cross-entropy. The transpose arises from the chain rule and Euclidean adjoint identity. It is generally not an inverse. The weight gradient is a sum of outer products with rank at most one per sample. Source: Goodfellow et al., Deep Learning Chapter 6, https://www.deeplearningbook.org/contents/mlp.html

## Slide 8: Independent gradient verification

Numerical and analytical gradients agree away from ReLU kinks. Other tests cover extreme logits, stratified splitting, affine collapse including biases, spectral identities, the convex hull witness and checkpoint restoration. Source: Stanford CS231n gradient checking, https://cs231n.github.io/neural-networks-3/

## Slide 9: Training and validation

Metrics at epoch k describe exactly k completed updates. The dashed line marks the selected checkpoint. The full trajectory can contain loss increases. The model restores the selected checkpoint, which may precede the final epoch. Test data are absent from the optimization curves.

## Slide 10: Primary results on disjoint data

All rows use initialization seed 42 and the same dataset split and update budget. Train, validation and test accuracies are clearly separate. Differences also reflect parameter count and optimization. This is not a capacity-matched study.

## Slide 11: Initialization sensitivity

The mean and sample standard deviation describe sensitivity to random initialization on this one split. We report every run, without selecting the best test seed. Sampling uncertainty, new data splits and distribution shift remain outside this study.

## Slide 12: The learned decision boundaries

The ReLU model produces piecewise affine logits. Its binary equal-logit boundary is piecewise linear. The comparison shows a learned winding boundary, not proof that every deep network succeeds or that every shallow network fails. The color surface does not prove probability calibration.

## Slide 13: Hidden geometry through PCA

SVD of centered training activations gives a PCA basis. We project test activations using the same training mean and basis. The plots state retained variance. These variance ratios concern activation matrices, not weight matrices. The output boundary is the diagonal z0=z1.

## Slide 14: Weight spectra and nullspaces

A 2-by-16 full-row-rank output map still has input nullity 14. The singular value ratio concerns nonzero directions, not an inverse on all input dimensions. SVD factors need not be unique. Full weight rank does not prove every neuron is active or that gradients are stable. Source: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html

## Slide 15: SVD approximation of the hidden map

The dense W2 has 256 entries. A rank k factorization would store 32k entries, excluding biases. Our measurements materialize the approximation and do not save compressed factors. The discarded singular-value energy gives the optimal Frobenius error, but classification accuracy can change. This directly extends SVD compression from supplied Project 14.

## Slide 16: Conclusions and limits

Conclude with a precise claim: the chosen configuration learns a nonlinear boundary, and independent numerical checks support the gradient implementation. Do not claim linear algebra alone creates AI, that depth is mandatory, or that weight spectra prove data-manifold recovery. A sufficiently wide shallow nonlinear network can be expressive. Source: Goodfellow et al., Deep Learning Chapter 6. Read docs/REPORT.md and docs/VIVA_GUIDE.md for details.

# Viva guide

**UE25MA242A: Mathematical Foundation for AI & Data Science**

## What is the project actually demonstrating?

A NumPy network learns a nonlinear boundary on synthetic spirals using affine maps, ReLU, matrix gradients and gradient descent. We inspect hidden representations with PCA and weight maps with SVD. Linear algebra is essential, alongside calculus, probability and optimization.

## How does it connect to the supplied PDF and ZIP?

The problem sheet lists transformations, PCA and SVD compression as Projects 10, 11 and 14. We extend those concepts to neural network representations and SVD approximation of a learned weight matrix. Its acceptance as an extension depends on the instructor's requirements.

## Why is each layer affine?

The bias translates the output: `z = W a + b`. A linear map must send zero to zero. An affine layer need not.

## Why do we need ReLU?

Without hidden nonlinearities, compositions of affine maps collapse to another affine map. ReLU adds piecewise behavior and makes the input-space boundary more expressive. Two hidden layers are useful in this configuration, but depth is not universally mandatory. A sufficiently wide shallow nonlinear network can also be expressive.

## How many parameters are there?

`16*2+16 = 48`, `16*16+16 = 272`, `2*16+2 = 34`, giving **354** total. There are 320 weights and 34 biases. The original draft's 374 was an arithmetic error.

## Why does the transpose appear in backpropagation?

By the chain rule, the jth input gradient is `sum_i W[i,j]*delta[i]`. In vector form it is `W.T @ delta`. This is the Euclidean adjoint action. It is not generally an inverse.

## Where is the averaging factor?

The output error `P-Y` is unaveraged. Divide `delta @ A.T` and the bias sum by the number of training samples exactly once. Averaging both delta and the parameter gradient would incorrectly divide twice.

## How do you know the gradients are correct?

Compare every scalar derivative in a small network against centered finite differences. Avoid perturbations crossing ReLU kinks. Also check an identity-activation network and independently verify stable loss, splitting and spectral identities.

## Why is 90% called test accuracy?

It is 72 correct predictions on 80 held-out test points. Those points do not influence gradient updates or checkpoint selection. Training accuracy is separately 97.08%. Validation accuracy is 93.75%, and validation loss selects the checkpoint.

## Why can the linear baseline exceed 50%?

Nonseparability rules out perfect classification, not all improvement over chance. Our witness records intersecting segments within opposite classes. Their convex hulls overlap, ruling out strict perfect affine separation. It does not imply a 50% ceiling.

## Why did the shallow network perform poorly?

We tested one hidden width and one fixed optimization configuration. It is an empirical result for that model under this budget, not a theorem about every shallow network. Width, initialization and learning rate may change the outcome. Models also have different parameter counts.

## Does a high condition number prove noise suppression?

No. It describes anisotropic amplification by a weight map. ReLU masks, activation distributions and the readout affect the complete network. We would need separate perturbation experiments to establish noise robustness.

## Does full rank mean there are no inactive neurons?

No. Rank describes linear independence in the weight map. ReLU can zero a neuron's activation even if the matrix has full rank. The 2-by-16 output map has input nullity 14 despite full row rank.

## Is SVD unique? Are its factors rotations?

Factors can vary through paired sign flips or repeated singular values. Orthogonal basis changes can include reflections. Rectangular thin factors span retained input and output subspaces rather than every coordinate.

## How do weight SVD and PCA differ?

Weight SVD decomposes one weight matrix. PCA decomposes centered activation data. Weight singular values describe amplification by a linear map. Activation singular values describe variance of observed representations. We fit PCA only on training activations and report variance retained by two components.

## Why can compression change accuracy even if it retains most energy?

SVD minimizes matrix approximation error at a fixed rank. Classification depends on decision margins and downstream transformations, so retained weight energy does not guarantee identical predictions. Rank-4 W2 retains 83.11% of weight energy but gives 77.50% test accuracy. Rank 8 retains 95.61% of energy and matches the primary run's 90.00% test accuracy.

## What are the main limitations?

There are 400 synthetic points, one split and three initializations. Deep test results vary from 78.75% to 92.50%. Architectures are not matched for parameter count. We have not evaluated real-world generalization, probability calibration or a full validation-only hyperparameter search.

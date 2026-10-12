# Neural Networks through Linear Algebra

> **Course**: UE25MA242A — Mathematical Foundation for AI & Data Science (PES University)

## What is this project?
This project demonstrates how a **neural network learns using pure Linear Algebra** — implemented completely from scratch in **NumPy** without PyTorch, TensorFlow, or Scikit-learn.

* **The Problem**: A purely linear classifier (matrix multiplication $W x + b$) can only draw straight decision lines and completely fails to separate intertwined spirals (~50% accuracy).
* **The Solution**: Adding simple non-linear activations ($\text{ReLU}$) allows stacks of affine transformations to bend and twist decision boundaries, reaching **90%+ accuracy**.
* **Linear Algebra Concepts Used**:
  * **Matrix Calculus & Adjoint Transposes**: Forward and backward propagation implemented with explicit matrix products ($W \cdot A + b$) and transposes ($W^T \delta$).
  * **PCA (Principal Component Analysis)**: Projecting 16-dimensional hidden neuron activations down to 2D to visualize how data unfolds across layers.
  * **SVD (Singular Value Decomposition)**: Inspecting directional amplification of weights and testing low-rank weight matrix compression.

---

## Quick Start: How to Run Locally

### 1. Setup Environment
Clone or navigate to the project directory, then create a virtual environment:

```bash
cd neural_network_LA_project
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Live Demo
Watch the network train in your terminal and view the learned decision boundary:

```bash
./run_live_demo.sh
```

### 3. Run Experiments & Tests
To retrain models and regenerate all figures and metrics:
```bash
python src/run_project.py
```

To run independent numerical and gradient checks:
```bash
python -m unittest discover -s tests -v
```

---

## Key Results

| Model | Parameters | Train Acc | Test Acc | Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Linear** ($2 \to 2$) | 6 | 50.0% | 58.8% | Fails — bounded by linear non-separability |
| **Shallow** ($2 \to 16 \to 2$) | 82 | 57.5% | 58.8% | Underfits under vanilla gradient descent |
| **Deep ReLU** ($2 \to 16 \to 16 \to 2$) | 354 | **97.1%** | **90.0%** | Successfully learns the winding spiral boundary |

*Evaluated across 3 random seeds: test accuracy ranges from 78.8% to 92.5% (mean 87.1%).*

---

## Key Files & Submission Artifacts

* **Report**: [`PROJECT_REPORT.pdf`](neural_network_LA_project/docs/PROJECT_REPORT.pdf) — Comprehensive 9-page formal report.
* **Presentation**: [`Neural_Network_Linear_Algebra.pptx`](neural_network_LA_project/presentation/Neural_Network_Linear_Algebra.pptx), [`Neural_Network_Linear_Algebra.pdf`](neural_network_LA_project/presentation/Neural_Network_Linear_Algebra.pdf), & [`Presentation_Script.txt`](neural_network_LA_project/presentation/Presentation_Script.txt).
* **Jupyter Notebook**: [`neural_network_project.ipynb`](neural_network_LA_project/notebooks/neural_network_project.ipynb) — Step-by-step interactive walkthrough.
* **Core Code**:
  * [`neural_network.py`](neural_network_LA_project/src/neural_network.py): Pure NumPy network, SVD analysis, and gradient checks.
  * [`run_project.py`](neural_network_LA_project/src/run_project.py): Full experiment pipeline and metrics generator.
  * [`test_math.py`](neural_network_LA_project/tests/test_math.py): Mathematical identity unit tests.

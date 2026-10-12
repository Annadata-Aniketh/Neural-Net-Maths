"""Generate the report, notebook and presentation content from metrics.json.

Run with Python plus reportlab. Experiment dependencies remain NumPy,
Matplotlib and Pillow. The PPTX uses the separate build_slides.mjs script.
"""
from pathlib import Path
from html import escape
import json
import math
import csv

ROOT = Path(__file__).resolve().parents[1]
REFS = [
    ("PES University, Mini-Project Problem Statements, supplied PDF, one page.", "../../references/mini_project_problem_statements.pdf"),
    ("Anna Zemlyanova, Applied projects for an introductory linear algebra class, supplied adapted manual. Projects 10, 11 and 14. CC BY-NC 4.0.", "../../references/applied_linear_algebra_manual.pdf"),
    ("Goodfellow, Bengio and Courville, Deep Learning (2016), Chapter 6, Deep Feedforward Networks.", "https://www.deeplearningbook.org/contents/mlp.html"),
    ("Stanford CS231n, Neural Networks Part 3, gradient checking.", "https://cs231n.github.io/neural-networks-3/"),
    ("NumPy documentation, numpy.linalg.svd.", "https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html"),
]


def wilson(correct, n):
    p, z = correct/n, 1.96
    center = (p + z*z/(2*n))/(1 + z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return center-half, center+half


def build():
    data = json.loads((ROOT / "results/metrics.json").read_text())
    pca = json.loads((ROOT / "results/pca.json").read_text())
    primary = {r["model"]: r for r in data["runs"] if r["seed"] == 42}
    deep = primary["deep"]
    check = data["gradient_check"]
    low, high = wilson(deep["test"]["correct"], deep["test"]["samples"])
    percent = lambda x: f"{100*x:.2f}%"
    row_table = [["Model", "Parameters", "Train", "Validation", "Test", "Epoch"]] + [
        [name.capitalize(), str(r["parameters"]), percent(r["train"]["accuracy"]),
         percent(r["validation"]["accuracy"]), percent(r["test"]["accuracy"]), str(r["selected_epoch"])]
        for name, r in primary.items()]
    seed_table = [["Model", "Seed 42", "Seed 43", "Seed 44", "Mean", "Sample SD"]] + [
        [name.capitalize()] + [percent(r["test"]["accuracy"]) for r in data["runs"] if r["model"] == name]
        + [percent(data["seed_summary"][name]["mean_test_accuracy"]),
           f"{100*data['seed_summary'][name]['std_test_accuracy']:.2f} pp"] for name in primary]
    spectral_table = [["Weight", "Shape", "Rank", "Input nullity", "s_max / s_min"]] + [
        [f"W{d['layer']}", " x ".join(map(str, d["shape"])), str(d["rank"]), str(d["input_nullity"]),
         f"{d['condition_nonzero_spectrum']:.3f}"] for d in data["spectral_analysis"]]
    compression_table = [["Rank k", "Energy", "Factor entries", "Validation", "Test"]] + [
        [str(r["rank"]), percent(r["weight_energy_retained"]), str(r["factor_entries"]),
         percent(r["validation"]["accuracy"]), percent(r["test"]["accuracy"])] for r in data["rank_truncation"]]
    pages = [
        {"title": "Neural networks through matrix operations", "subtitle": "Affine maps, backpropagation, PCA and SVD on two spirals",
         "paragraphs": [
             "UE25MA242A: Mathematical Foundation for AI & Data Science | Mini-project | Reproducible NumPy implementation",
             "Research question: How does adding ReLU to compositions of affine maps change a classifier's decision boundary, and what can matrix decompositions tell us about the learned maps?",
             f"We implement a 2-16-16-2 network with 354 trainable scalar parameters. We train on 240 synthetic points, select a checkpoint using 80 validation points, and evaluate on 80 separate test points. The primary run (initialization seed 42) classifies {deep['test']['correct']} of 80 test points correctly ({percent(deep['test']['accuracy'])}).",
             "The experiment includes a linear baseline and a network with one hidden layer. All models use explicit NumPy gradients. Finite differences independently check the derivatives. PCA visualizes hidden activations, and SVD provides weight spectra and a controlled rank truncation experiment.",
             "Linear algebra supplies the matrix operations. Nonlinear activation, probability, differential calculus and optimization supply additional essential ingredients. This project demonstrates their interaction."],
         "image": "model_comparison.png", "caption": "Figure 1. Seed 42 models at their validation-selected checkpoints. Points are test samples."},
        {"title": "1. Inspiration, dataset and separation", "paragraphs": [
             "The supplied problem sheet lists 14 applications. This project extends Project 10 (matrix transformations), Project 11 (PCA) and Project 14 (SVD compression). It is an inspired extension, rather than a literal implementation of a listed neural network problem. Confirm that the instructor accepts extensions before submission.",
             "The confirmed course code is UE25MA242A, matching the supplied sheet's header. The adapted manual mixes Python and MATLAB syntax. We use verified NumPy syntax and cite its mathematical topics rather than copying its code.",
             "For each class c in {0,1}, generate 200 points with theta uniformly spaced from 0 to 4*pi and radius theta/(4*pi). Add independent Cartesian Gaussian noise with standard deviation 0.0125 to each coordinate. A local random generator uses data seed 42. This creates two turns per arm, with opposite arms shifted by pi.",
             "A stratified split with seed 2026 assigns 120/40/40 points per class to training/validation/test. We do not normalize these already bounded coordinates. Every sample is a column of X. Synthetic test points assess generalization to held-out points from this generator, not to real-world data.",
             "For a two-class affine softmax classifier, equal class scores define a line. Three failed candidate lines alone would not prove nonseparability. Instead, our code records two intersecting within-class segments. Their intersection belongs to both class convex hulls. A strict affine separator would have to assign the same point opposite signs, which is impossible. This rules out perfect affine classification, but does not impose a 50% accuracy ceiling."],
         "equations": ["x_c(theta) = (r*cos(theta+c*pi), r*sin(theta+c*pi))^T + epsilon", "epsilon ~ N(0, 0.0125^2 I_2),  r = theta/(4*pi)", "(w_1-w_0)^T x + (b_1-b_0) = 0"],
         "image": "dataset_split.png", "caption": "Figure 2. Disjoint training, validation and test samples."},
        {"title": "2. Forward propagation and dimensions", "paragraphs": [
             "We use column-oriented samples and three trainable affine maps, with two ReLU hidden layers. A layer computes Z = WA + b*1^T. The bias makes the map affine. ReLU acts coordinate by coordinate, and the final layer uses softmax to convert logits into class probabilities.",
             "Layer 1 has W1 in R^(16 x 2) and b1 in R^(16 x 1), giving 32+16=48 parameters. Layer 2 has W2 in R^(16 x 16) and b2 in R^(16 x 1), giving 256+16=272. Layer 3 has W3 in R^(2 x 16) and b3 in R^(2 x 1), giving 32+2=34. Total: 354 parameters, comprising 320 weights and 34 biases.",
             "During training m=240, so Z1 and A1 have shape (16,240), Z2 and A2 have shape (16,240), and Z3 and P have shape (2,240). The same equations work with m=80 during evaluation. Broadcasting b across columns implements b*1^T.",
             "Hidden weights use independent normal samples with variance 2/fan_in (He scaling). The output uses variance 1/fan_in. Biases start at zero. These are initialization heuristics based on approximate moment preservation, not guarantees about singular values or training stability.",
             "If hidden activations are identities, composing affine layers gives another affine map, including an effective bias. ReLU breaks that collapse and produces a piecewise affine logit function. Within each fixed activation region, the binary decision boundary is a line segment. Many such regions can approximate a winding boundary. Softmax preserves which logit is largest."],
         "equations": ["A0 = X,  Z_l = W_l A_(l-1) + b_l 1^T", "A1 = max(0,Z1),  A2 = max(0,Z2),  P = softmax(Z3)", "W_eff = W3 W2 W1", "b_eff = W3 W2 b1 + W3 b2 + b3"],
         "image": "weight_heatmaps.png", "caption": "Figure 3. Learned weights, using separate color scales for the three maps."},
        {"title": "3. Loss and matrix backpropagation", "paragraphs": [
             "The objective is mean cross-entropy over the training columns. Subtracting each column's largest logit before computing log-sum-exp avoids overflow. We compute loss from logits, so even extremely wrong predictions have a finite meaningful loss without clipping probabilities. Cross-entropy uses natural logarithms.",
             "Let D_l denote the unaveraged logit error per sample. For softmax combined with cross-entropy, D3=P-Y. Hidden errors multiply by the next weight transpose and the elementwise ReLU mask. Parameter gradients average over m exactly once.",
             "The transpose follows directly from the chain rule: the derivative of z_i with respect to a_j is W_ij, so the previous-layer gradient has component sum_i W_ij D_i. In Euclidean inner products, this is W^T D, the adjoint action. It is generally not the inverse of W.",
             "D_l A_(l-1)^T is a sum of per-sample outer products. Each contribution has rank at most one, and the batch sum need not have rank one. The bias derivative sums errors over columns. Full-batch gradient descent subtracts learning_rate times each gradient from its parameter.",
             f"A centered finite-difference check on a small 2-4-3-2 model checked {check['checked']} scalar entries, skipping {check['skipped_kinks']} ReLU kink crossings. Maximum absolute error: {check['max_absolute_error']:.2e}. Separate tests cover an identity-activation model, stable extreme-logit loss, affine collapse with biases, splitting and spectral identities. ReLU is nondifferentiable at zero, where the implementation chooses derivative zero."],
         "equations": ["L = -(1/m) sum_j sum_c Y_cj log(P_cj)", "D3 = P-Y", "D_l = (W_(l+1)^T D_(l+1)) * 1[Z_l > 0]", "grad(W_l) = D_l A_(l-1)^T / m", "grad(b_l) = D_l 1 / m", "theta_next = theta - 0.1 * grad(theta)", "finite_difference_i = (L(theta+h*e_i)-L(theta-h*e_i))/(2*h)" ]},
        {"title": "4. Evaluation and training evidence", "paragraphs": [
             "We declare one data seed, one split seed, initialization seeds 42/43/44, 12,000 full-batch updates, and learning rate 0.1. Models share the split and update budget. Every 50 completed updates, we record training and validation loss and accuracy. We retain the checkpoint with smallest recorded validation loss. Test data never enter gradient computation or checkpoint selection.",
             "Epoch 0 represents initialization. Metrics at epoch k describe parameters after exactly k updates. The selected checkpoint can precede the final epoch. Learning curves show the entire training trajectory, while reported evaluation uses the restored selected checkpoint. Loss need not decrease monotonically at this finite learning rate.",
             "Primary results (seed 42) follow. Differences between the linear, shallow and deep models reflect both architecture and parameter count. The comparison is not controlled for equal capacity or equally optimized hyperparameters.",
             f"The main deep network gets {deep['test']['correct']}/80 test predictions correct. An approximate Wilson 95% binomial interval is {percent(low)} to {percent(high)}. This interval describes sampling uncertainty only approximately because the radial grid is structured and not an independent random sample of every possible real-world input.",
             "We also report all three initialization runs on the same split. The sample standard deviation describes initialization sensitivity only. Three seeds and one split do not characterize all sources of uncertainty. The main run is declared as seed 42, not whichever run scores highest on test data."],
         "table": row_table, "table2": seed_table,
         "image": "learning_curves.png", "caption": "Figure 4. Training and validation curves. The dashed line marks the restored checkpoint."},
        {"title": "5. Geometry and PCA projections", "paragraphs": [
             "Hidden layers live in R^16, so a two-dimensional scatterplot cannot display the entire representation. For each hidden layer, we center the training activation matrix H, compute its thin SVD, and use the first two left singular vectors as a PCA basis. We project test activations using that same training mean and basis.",
             f"The first two components retain {percent(pca[0]['training_variance_retained'])} of training activation variance for hidden layer 1 and {percent(pca[1]['training_variance_retained'])} for hidden layer 2. The ratios use squared singular values of centered activations. These are activation variance ratios, distinct from weight-matrix energy fractions.",
             "A PCA plot may discard classification-relevant directions. Overlap in two projected components does not prove the full hidden representation is inseparable, and visible separation does not prove perfect generalization. We label every plot as a projection and include its retained variance.",
             "In the two-logit output plot, the decision boundary is z0=z1, a diagonal line. Each point's side predicts a class. This readout is linear in hidden coordinates while the input-space boundary is piecewise linear because of ReLU. The softmax probability surface can be smooth within regions, but it is not evidence of probability calibration.",
             "ReLU can merge distinct inputs and kill coordinates. A learned hidden transformation therefore need not be invertible or a homeomorphism. Full rank of a weight matrix says nothing by itself about the number of active neurons on the dataset."],
         "equations": ["H_centered = H_train - mean_train 1^T = U S V^T", "test_projection = U_two^T (H_test - mean_train 1^T)", "retained_variance = (s1^2+s2^2) / sum_i s_i^2"],
         "image": "hidden_representations.png", "caption": "Figure 5. PCA fits use training data. Colored points are test activations."},
        {"title": "6. SVD, rank and Gram eigenvalues", "paragraphs": [
             "For W in R^(p x q), thin SVD gives W=U diag(s) V^T with r=min(p,q) singular values. The columns of U and V are orthonormal on their retained subspaces. They encode changes of orthonormal basis, possibly involving reflections. SVD factors are not unique, for example because paired singular vectors can change sign.",
             "Singular values measure directional amplification of a weight map before its activation. Squared singular values are the nonzero eigenvalues of W^T W. If q>r, the Gram matrix also contains q-r structural zeros. We compare eigvalsh(W^T W) with this correctly zero-padded spectrum and check SVD reconstruction independently.",
             "A full-row-rank 2 x 16 output map has a 14-dimensional input nullspace. Its two returned singular values can have a modest ratio, even though the map cannot be inverted on all of R^16. We therefore label the ratio s_max/s_min as conditioning on the nonzero singular spectrum. Rank does not imply an orthogonal projection or prevent information loss.",
             "Weight conditioning describes anisotropy of each affine map. It does not establish noise removal, gradient stability, neuron activity or probability calibration. The network Jacobian also depends on ReLU masks. A product of spectral norms bounds its operator norm, but such a bound alone does not show that gradients stayed away from zero.",
             "Spectral diagnostics below refer to the seed 42 selected deep checkpoint. Every value comes from saved parameter arrays."],
         "table": spectral_table,
         "equations": ["W = U diag(s) V^T", "eigenvalues(W^T W) = {s_i^2} plus structural zeros", "rank(W) + nullity(W) = number of input coordinates"],
         "image": "singular_values.png", "caption": "Figure 6. Singular values of the individual weight maps."},
        {"title": "7. Rank truncation and compression", "paragraphs": [
             "Project 14 compresses images using SVD. We extend the same idea to the learned 16 x 16 hidden weight matrix W2. Keep only its k largest singular values, set W2_k=U_k diag(s_k) V_k^T, and keep every other weight and bias fixed. We evaluate the prescribed ranks 2,4,8,12,16 without retraining or choosing ranks from test accuracy.",
             "The truncated SVD minimizes Frobenius error among matrices of rank at most k. The error equals the square root of the sum of discarded squared singular values. We numerically verify that identity. This is an approximation guarantee for W2, not for the whole classifier's predictions.",
             "A factorized implementation could store B=U_k diag(s_k) and C=V_k^T, computing B(CA), using 32k scalar weight entries instead of 256. Rank 4 would halve this layer's weight storage, while rank 8 would save none. Bias storage remains unchanged. Our experiment materializes W2_k for direct comparison, so the saved model arrays themselves are not compressed factor files.",
             "The table separates retained Frobenius energy from validation and test accuracy. Preserving most weight energy need not preserve all decision boundaries. Conversely, a change can sometimes improve an individual held-out score. Neither observation proves that training found a low-dimensional data manifold.",
             "This experiment demonstrates a concrete application of SVD to model approximation. It gives a defensible link to the assigned material without claiming that the classifier is inherently low rank."],
         "table": compression_table,
         "equations": ["W2_k = U_k diag(s_k) V_k^T", "||W2-W2_k||_F^2 = sum_(i>k) s_i^2", "energy_retained(k) = sum_(i<=k) s_i^2 / sum_i s_i^2"]},
        {"title": "8. Conclusions, limits and reproduction", "paragraphs": [
             "The main experiment shows that this ReLU network learns a winding decision boundary using explicit matrix operations and gradients. The linear baseline has a limited boundary family. The shallow baseline's poor result under this particular width, learning rate and budget does not prove that every shallow network must fail. A sufficiently wide nonlinear hidden layer can be expressive, while successful optimization remains a separate question [3].",
             "The dataset is small, balanced and synthetic. We used one split and three initializations, no systematic hyperparameter search and no real-world distribution shift. Near-origin noise can make class distinctions difficult. More data, additional predefined splits and a broader validation-only architecture study would strengthen the empirical comparison.",
             "Reproduction from the project directory: create a virtual environment, install requirements.txt, then run python src/run_project.py. Run python -m unittest discover -s tests -v for numerical checks. The script regenerates dataset.npz, selected weights, CSV histories, metrics.json and figures. The notebook imports the same implementation, so it cannot diverge into a second unverified training algorithm.",
             "The presentation, notebook and writeup read the generated metrics. Add your team names before submission. Original reference documents remain unchanged under the workspace references directory. The source ZIP is retained without unpacking unrelated image databases.",
             "References below attribute the mathematical inspiration and implementation checks. The adapted lab manual contains formatting and programming errors, so it should not serve as an unquestioned Python reference."],
         "references": REFS},
    ]
    report_md = ["# " + pages[0]["title"], "\n" + pages[0]["subtitle"]]
    def md_table(table):
        return "\n".join(["| " + " | ".join(table[0]) + " |", "| " + " | ".join(["---"]*len(table[0])) + " |"]
                         + ["| " + " | ".join(row) + " |" for row in table[1:]])
    for i, page in enumerate(pages):
        if i:
            report_md.append("\n## " + page["title"])
        report_md.extend("\n"+p for p in page["paragraphs"])
        for equation in page.get("equations", []):
            report_md.append("\n```text\n"+equation+"\n```")
        for field in ("table", "table2"):
            if field in page:
                report_md.append("\n"+md_table(page[field]))
        if page.get("image"):
            report_md.append(f"\n![{page['caption']}](../results/figures/{page['image']})")
        for index, (label, url) in enumerate(page.get("references", []), 1):
            report_md.append(f"\n{index}. [{label}]({url})")
    (ROOT / "docs/REPORT.md").write_text("\n".join(report_md)+"\n")
    make_pdf(pages)
    make_notebook(primary, pca)
    make_slides(data, primary, row_table, seed_table, spectral_table, compression_table, check)


def make_pdf(pages):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.pagesizes import A4
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Body", fontName="Helvetica", fontSize=9.7, leading=14.1, spaceAfter=8, textColor=colors.HexColor("#263b33")))
    styles.add(ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=21, leading=26, spaceAfter=14, textColor=colors.HexColor("#183c2d")))
    styles.add(ParagraphStyle("Equation", fontName="Courier", fontSize=9, leading=13, leftIndent=10, spaceAfter=5))
    styles.add(ParagraphStyle("CaptionSmall", fontName="Helvetica", fontSize=8.3, leading=11, spaceAfter=8, textColor=colors.HexColor("#526a5c")))
    width, height = A4
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#aac2b0"))
        canvas.line(42, 35, width-42, 35)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526a5c"))
        canvas.drawString(42, 22, "Neural networks through matrix operations | NumPy mini-project")
        canvas.drawRightString(width-42, 22, str(doc.page))
    story = []
    for index, page in enumerate(pages):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(escape(page["title"]), styles["Heading"]))
        if page.get("subtitle"):
            story.append(Paragraph(escape(page["subtitle"]), styles["Heading2"]))
        story.extend(Paragraph(escape(p), styles["Body"]) for p in page["paragraphs"])
        for equation in page.get("equations", []):
            story.append(Paragraph(escape(equation), styles["Equation"]))
        for field in ("table", "table2"):
            if field in page:
                table = page[field]
                cells = [[Paragraph(escape(str(v)), styles["CaptionSmall"]) for v in row] for row in table]
                t = Table(cells, colWidths=[(width-84)/len(table[0])]*len(table[0]), repeatRows=1, hAlign="LEFT")
                t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e5eee6")),
                                      ("LINEBELOW", (0,0), (-1,0), .7, colors.HexColor("#526a5c")),
                                      ("LINEBELOW", (0,1), (-1,-1), .25, colors.HexColor("#c5d7c8")),
                                      ("VALIGN", (0,0), (-1,-1), "TOP"),
                                      ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
                story.extend((t, Spacer(1, 10)))
        if page.get("image"):
            path = ROOT / "results/figures" / page["image"]
            iw, ih = ImageReader(str(path)).getSize()
            max_height = 174 if index != 0 else 235
            scale = min((width-84)/iw, max_height/ih)
            story.append(KeepTogether([Image(str(path), width=iw*scale, height=ih*scale),
                                      Paragraph(escape(page["caption"]), styles["CaptionSmall"])]))
        for number, (label, url) in enumerate(page.get("references", []), 1):
            if url.startswith("http"):
                label += " " + url
            story.append(Paragraph(escape(f"[{number}] {label}"), styles["CaptionSmall"]))
    doc = SimpleDocTemplate(str(ROOT / "docs/PROJECT_REPORT.pdf"), pagesize=A4,
                           rightMargin=42, leftMargin=42, topMargin=40, bottomMargin=46,
                           title=pages[0]["title"], author="Maths mini-project team")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def make_notebook(primary, pca):
    cells = []
    def md(s):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
    def code(s):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)})
    md("# Neural networks through matrix operations\n\n**UE25MA242A: Mathematical Foundation for AI & Data Science**\n\nThis notebook explains and inspects the reproducible experiment. Read `docs/REPORT.md` for the complete derivation. It imports the shared NumPy implementation. Run all cells from the project or notebooks directory.\n\nCourse inspiration: supplied Projects 10 (transformations), 11 (PCA) and 14 (SVD). Neural networks are an extension of these topics.\n")
    code("from pathlib import Path\nimport sys, json\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif ROOT.name == 'notebooks':\n    ROOT = ROOT.parent\nif not (ROOT / 'src/neural_network.py').exists():\n    raise RuntimeError('Open this notebook from neural_network_LA_project or its notebooks directory.')\nsys.path.insert(0, str(ROOT / 'src'))\nfrom neural_network import NeuralNetwork, spiral_data, stratified_split, gradient_check, spectral_analysis, separation_witness\nfrom run_project import run\n")
    md("## 1. Reproducible data\n\nSamples are columns: $X\\in\\mathbb{R}^{2\\times 400}$. Each spiral has two turns and Cartesian noise standard deviation 0.0125. The disjoint stratified split has 240 training, 80 validation and 80 test points.\n")
    code("X, Y = spiral_data(seed=42)\ntrain, validation, test = stratified_split(Y, seed=2026)\nassert len(set(np.concatenate([train, validation, test]))) == 400\nprint(X.shape, Y.shape, [len(i) for i in (train, validation, test)])\nprint('Convex hull overlap witness:', separation_witness(X,Y))\n")
    md("## 2. Shared forward and backward implementation\n\n$Z_l=W_l A_{l-1}+b_l\\mathbf{1}^{T}$, hidden $A_l=\\max(0,Z_l)$ and output $P=\\mathrm{softmax}(Z_3)$. The 2-16-16-2 model has $48+272+34=354$ parameters.\n\nFor mean cross-entropy, use unaveraged $D_3=P-Y$, $D_l=(W_{l+1}^{T}D_{l+1})\\odot\\mathbf{1}_{Z_l>0}$, $\\nabla W_l=D_l A_{l-1}^{T}/m$, $\\nabla b_l=D_l\\mathbf{1}/m$. Averaging occurs exactly once.\n")
    code("model = NeuralNetwork(seed=42)\nP, cache = model.forward(X[:,train])\ngrads = model.backward(Y[:,train], cache)\nprint('Parameters:', model.parameter_count)\nprint('Gradient shapes:', {k:v.shape for k,v in grads.items()})\nsmall = NeuralNetwork((2,4,3,2), seed=17)\nprint('Centered finite differences:', gradient_check(small, X[:,train[:5]], Y[:,train[:5]]))\n")
    md("## 3. Load or regenerate the declared experiment\n\nBy default use the saved results. Set `RETRAIN=True` to run all nine trainings with the declared seeds and budget. Checkpoint selection uses validation loss, and test predictions have no effect on selection.\n")
    code("RETRAIN = False\nif RETRAIN or not (ROOT / 'results/metrics.json').exists():\n    run(ROOT / 'results')\nmetrics = json.loads((ROOT / 'results/metrics.json').read_text())\nfor row in metrics['runs']:\n    print(row['model'], 'seed', row['seed'], 'epoch', row['selected_epoch'],\n          'train', round(row['train']['accuracy'],4),\n          'validation', round(row['validation']['accuracy'],4),\n          'test', round(row['test']['accuracy'],4))\nwith np.load(ROOT / 'results/weights_deep_42.npz') as saved:\n    model.params = {k:saved[k].copy() for k in saved.files}\nprint('Restored test evaluation:', model.evaluate(X[:,test],Y[:,test]))\n")
    code("for filename in ('model_comparison.png','learning_curves.png','confusion_matrix.png'):\n    display(Image(filename=str(ROOT / 'results/figures' / filename)))\n")
    md("## 4. PCA and SVD answer different questions\n\nPCA fits a basis to **centered training activations** and projects test activations with that basis. A two-dimensional plot may lose classification information. Weight SVD describes a single linear map before ReLU. It cannot establish neuron activity, calibration or whole-network invertibility.\n\nFor a rectangular $2\\times16$ output matrix, full row rank is 2 and input nullity is 14. Gram eigenvalues include structural zeros.\n")
    code("for d in spectral_analysis(model):\n    print('Layer',d['layer'], 'shape',d['shape'], 'rank',d['rank'],\n          'input nullity',d['input_nullity'], 'nonzero-spectrum ratio',d['condition_nonzero_spectrum'])\n    assert d['gram_identity_max_error'] < 1e-10\n    assert d['svd_reconstruction_relative_error'] < 1e-10\ndisplay(Image(filename=str(ROOT / 'results/figures/hidden_representations.png')))\ndisplay(Image(filename=str(ROOT / 'results/figures/singular_values.png')))\n")
    md("## 5. SVD rank approximation\n\n$W_{2,k}=U_k\\Sigma_kV_k^T$ and $\\|W_2-W_{2,k}\\|_F^2=\\sum_{i>k}\\sigma_i^2$. This guarantees optimal matrix approximation at rank k in Frobenius norm, not preserved classifier accuracy. The factorization uses 32k entries, compared with 256 for dense W2. The supplied results materialize W2_k and do not save compressed factors.\n")
    code("U,s,Vt = np.linalg.svd(model.params['W2'],full_matrices=False)\nfor row in metrics['rank_truncation']:\n    k = row['rank']\n    Wk = (U[:,:k]*s[:k]) @ Vt[:k]\n    error = np.linalg.norm(model.params['W2']-Wk,'fro')\n    assert np.isclose(error,np.linalg.norm(s[k:]),atol=1e-12)\n    print('rank',k, 'energy',round(row['weight_energy_retained'],4),\n          'factor entries',row['factor_entries'], 'test',row['test']['accuracy'])\n")
    md("## Conclusions and limits\n\nThe experiment supports the benefit of nonlinear features under this configuration. It does not prove that depth is mandatory or that linear classifiers are limited to 50%. One hidden nonlinear layer can be expressive at sufficient width. Three initialization seeds and one synthetic split are a limited empirical study.\n\nReferences: [Deep Learning, Chapter 6](https://www.deeplearningbook.org/contents/mlp.html), [Stanford gradient checks](https://cs231n.github.io/neural-networks-3/), [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html), and the supplied references preserved in the workspace.\n")
    for i, cell in enumerate(cells):
        cell["id"] = f"maths-{i:02}"
    notebook = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                             "language_info":{"name":"python","version":"3.14"}},"nbformat":4,"nbformat_minor":5}
    (ROOT / "notebooks/neural_network_project.ipynb").write_text(json.dumps(notebook,indent=2)+"\n")


def make_slides(data, primary, row_table, seed_table, spectral_table, compression_table, check):
    deep = primary["deep"]
    pct = lambda a: f"{100*a:.2f}%"
    slides = [
        {"title":"Neural networks through matrix operations", "body":"Affine maps, backpropagation, PCA and SVD\nUE25MA242A: Mathematical Foundation for AI & Data Science", "notes":"Introduce the research question: how do nonlinear activations extend affine classifiers, and how can matrix decompositions describe the learned maps? Our network uses NumPy for its calculations. It also relies on calculus, probability and optimization."},
        {"title":"Connection to the supplied projects", "body":"Project 10: compositions of matrix transformations\nProject 11: PCA of hidden activations\nProject 14: SVD and rank approximation\nAn extension of the supplied topics", "notes":"The sheet lists 14 projects, without a neural-network problem. Present this as an inspired extension and confirm that extensions are acceptable. The confirmed course code is UE25MA242A. Sources: supplied problem sheet and adapted Anna Zemlyanova manual, Projects 10, 11 and 14."},
        {"title":"Two spirals and a held-out test set", "body":"400 points, two turns per spiral\nCartesian noise standard deviation 0.0125\n240 training, 80 validation, 80 test\nBalanced classes in each split", "image":"dataset_split.png", "notes":"Data seed 42, split seed 2026. Every sample is a column of X. Test data do not participate in training or checkpoint selection. Coordinates already lie approximately within the unit disk."},
        {"title":"Why a line cannot classify every point", "body":"An affine classifier has a straight decision boundary\nTwo same-class segments intersect across classes\nThe intersection lies in both convex hulls\nPerfect strict separation is impossible", "notes":"Use the saved convex hull witness in metrics.json. An affine score preserves convex combinations, so the intersection could not have opposite signs. This proves impossibility of perfect classification on these points. It does not limit affine accuracy to 50%. The measured linear test score can exceed 50%."},
        {"title":"Architecture and parameter count", "table":[["Map","Weights","Biases","Parameters"],["Input to hidden 1","16 x 2","16 x 1","48"],["Hidden 1 to hidden 2","16 x 16","16 x 1","272"],["Hidden 2 to output","2 x 16","2 x 1","34"],["Total","320 entries","34 entries","354"]], "body":"Two ReLU hidden layers and a softmax output", "notes":"The architecture is 2-16-16-2 with three trainable affine maps. The original draft added the totals incorrectly. Hidden weights use He variance scaling and output weights use Xavier scaling. Initialization is a heuristic rather than a guarantee of spectral stability."},
        {"title":"Vectorized forward propagation", "body":"Z_l = W_l A_(l-1) + b_l 1^T\nA1 = ReLU(Z1), A2 = ReLU(Z2)\nP = softmax(Z3)\nEach column holds one sample", "notes":"For training, m=240. Z1 and Z2 have 16 rows, and Z3 has two rows. Bias broadcasting implements b times a row of ones. Without hidden nonlinearities, affine maps collapse to an effective W and b. Batch matrix computation eliminates loops over individual samples, but computational cost still grows with sample count and matrix size."},
        {"title":"Cross-entropy and matrix gradients", "body":"D3 = P - Y\nD_l = (W_(l+1)^T D_(l+1)) * 1[Z_l > 0]\ngrad(W_l) = D_l A_(l-1)^T / m\ngrad(b_l) = D_l 1 / m", "notes":"D is unaveraged, so divide by m exactly once in parameter gradients. The objective is mean cross-entropy. The transpose arises from the chain rule and Euclidean adjoint identity. It is generally not an inverse. The weight gradient is a sum of outer products with rank at most one per sample. Source: Goodfellow et al., Deep Learning Chapter 6, https://www.deeplearningbook.org/contents/mlp.html"},
        {"title":"Independent gradient verification", "body":f"Centered finite differences check every scalar\n{check['checked']} entries checked in a small network\nMaximum absolute error: {check['max_absolute_error']:.2e}\nSeven mathematical checks pass", "notes":"Numerical and analytical gradients agree away from ReLU kinks. Other tests cover extreme logits, stratified splitting, affine collapse including biases, spectral identities, the convex hull witness and checkpoint restoration. Source: Stanford CS231n gradient checking, https://cs231n.github.io/neural-networks-3/"},
        {"title":"Training and validation", "body":f"12,000 full-batch updates, learning rate 0.1\nCheckpoints every 50 updates\nSeed 42 selects epoch {deep['selected_epoch']}\nSelection minimizes validation loss", "image":"learning_curves.png", "notes":"Metrics at epoch k describe exactly k completed updates. The dashed line marks the selected checkpoint. The full trajectory can contain loss increases. The model restores the selected checkpoint, which may precede the final epoch. Test data are absent from the optimization curves."},
        {"title":"Primary results on disjoint data", "table":row_table, "body":f"Deep network: {deep['test']['correct']}/80 test predictions correct", "notes":"All rows use initialization seed 42 and the same dataset split and update budget. Train, validation and test accuracies are clearly separate. Differences also reflect parameter count and optimization. This is not a capacity-matched study."},
        {"title":"Initialization sensitivity", "table":seed_table, "body":"Three initializations on one fixed split", "notes":"The mean and sample standard deviation describe sensitivity to random initialization on this one split. We report every run, without selecting the best test seed. Sampling uncertainty, new data splits and distribution shift remain outside this study."},
        {"title":"The learned decision boundaries", "image":"model_comparison.png", "body":"Dots are held-out test points\nContours mark equal class probabilities", "notes":"The ReLU model produces piecewise affine logits. Its binary equal-logit boundary is piecewise linear. The comparison shows a learned winding boundary, not proof that every deep network succeeds or that every shallow network fails. The color surface does not prove probability calibration."},
        {"title":"Hidden geometry through PCA", "image":"hidden_representations.png", "body":"PCA bases use training activations\nTwo components can discard useful information", "notes":"SVD of centered training activations gives a PCA basis. We project test activations using the same training mean and basis. The plots state retained variance. These variance ratios concern activation matrices, not weight matrices. The output boundary is the diagonal z0=z1."},
        {"title":"Weight spectra and nullspaces", "table":spectral_table, "body":"W = U diag(s) V^T\nGram eigenvalues are s_i^2 plus structural zeros", "notes":"A 2-by-16 full-row-rank output map still has input nullity 14. The singular value ratio concerns nonzero directions, not an inverse on all input dimensions. SVD factors need not be unique. Full weight rank does not prove every neuron is active or that gradients are stable. Source: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html"},
        {"title":"SVD approximation of the hidden map", "table":compression_table, "body":"Fixed rank truncations of W2, without retraining", "notes":"The dense W2 has 256 entries. A rank k factorization would store 32k entries, excluding biases. Our measurements materialize the approximation and do not save compressed factors. The discarded singular-value energy gives the optimal Frobenius error, but classification accuracy can change. This directly extends SVD compression from supplied Project 14."},
        {"title":"Conclusions and limits", "body":"Nonlinear features improve this spiral experiment\nMatrix calculus makes learning explicit\nPCA and SVD describe different objects\nOne split and three seeds limit the conclusions", "notes":"Conclude with a precise claim: the chosen configuration learns a nonlinear boundary, and independent numerical checks support the gradient implementation. Do not claim linear algebra alone creates AI, that depth is mandatory, or that weight spectra prove data-manifold recovery. A sufficiently wide shallow nonlinear network can be expressive. Source: Goodfellow et al., Deep Learning Chapter 6. Read docs/REPORT.md and docs/VIVA_GUIDE.md for details."},
    ]
    (ROOT / "presentation/slides.json").write_text(json.dumps(slides,indent=2)+"\n")
    html = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Neural networks through matrix operations</title><style>',
            'body{margin:0;background:#142b20;color:#e9ede3;font-family:Arial,sans-serif} .slide{display:none;box-sizing:border-box;padding:5vh 5vw;min-height:100vh;background:#f2f1e8;color:#183c2d} .slide.active{display:flex;flex-direction:column;justify-content:center} h1{font-size:clamp(26px,4vw,56px);line-height:1.1;margin:0 0 3vh} p{white-space:pre-line;font-size:clamp(17px,2.2vw,29px);line-height:1.5;margin:2vh 0} img{max-width:100%;max-height:55vh;object-fit:contain;align-self:center} table{width:100%;border-collapse:collapse;font-size:clamp(14px,2vw,25px);margin:2vh 0} th,td{text-align:left;padding:1.7vh 1vw;border-bottom:1px solid #b6c6b4} th{background:#dce8dc} nav{position:fixed;bottom:12px;right:20px;display:flex;gap:10px;align-items:center;background:#183c2d;padding:8px;color:white} button{padding:8px 12px;font-size:14px;cursor:pointer} .folio{font-size:14px;margin-top:2vh} @media print{nav{display:none}.slide,.slide.active{display:flex;min-height:0;height:100vh;break-after:page;justify-content:center;padding:30px} @page{size:landscape;margin:0} img{max-height:50vh}}',
            '</style></head><body>']
    for i, slide in enumerate(slides):
        html.append(f'<section class="slide{" active" if i==0 else ""}" aria-label="Slide {i+1}"><h1>{escape(slide["title"])}</h1>')
        if slide.get("table"):
            html.append("<table>")
            for j, row in enumerate(slide["table"]):
                tag = "th" if j==0 else "td"
                html.append("<tr>"+"".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in row)+"</tr>")
            html.append("</table>")
        if slide.get("image"):
            html.append(f'<img src="../results/figures/{slide["image"]}" alt="{escape(slide["title"])}">')
        if slide.get("body"):
            html.append("<p>"+escape(slide["body"])+"</p>")
        html.append(f'<div class="folio">{i+1:02d} / {len(slides)}</div></section>')
    html.append('<nav aria-label="Presentation controls"><button id="previous">Previous</button><span id="counter">1 / 16</span><button id="next">Next</button><button id="print">Print</button></nav><script>let current=0;const slides=[...document.querySelectorAll(".slide")];function show(i){current=Math.max(0,Math.min(slides.length-1,i));slides.forEach((s,j)=>s.classList.toggle("active",j===current));document.querySelector("#counter").textContent=`${current+1} / ${slides.length}`;}document.querySelector("#previous").onclick=()=>show(current-1);document.querySelector("#next").onclick=()=>show(current+1);document.querySelector("#print").onclick=()=>print();document.addEventListener("keydown",e=>{if(["ArrowRight","PageDown"," "].includes(e.key)){e.preventDefault();show(current+1)}if(["ArrowLeft","PageUp"].includes(e.key)){e.preventDefault();show(current-1)}if(e.key==="Home")show(0);if(e.key==="End")show(slides.length-1)});</script></body></html>')
    (ROOT / "presentation/presentation.html").write_text("\n".join(html))
    script = ["# Presentation speaker notes", "\nThe deck has 16 slides. A 10-12 minute talk can spend 30-45 seconds on most slides and about a minute on gradients and results. Use your own words."]
    for i, slide in enumerate(slides,1):
        script.extend([f"\n## Slide {i}: {slide['title']}", "\n"+slide["notes"]])
    (ROOT / "docs/SPEAKER_NOTES.md").write_text("\n".join(script)+"\n")


if __name__ == "__main__":
    build()

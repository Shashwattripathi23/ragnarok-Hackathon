"""
Dummy data for testing the Study Guide interface.
All LLM responses follow the format:
  [ { "category": "...", "content": ... }, ... ]
"""

# ── Fake corpus files ──────────────────────────────────────────────
DUMMY_FILES = {
    "Lecture_01_Intro_to_ML.pdf": (
        "Machine Learning is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. Supervised learning uses labelled "
        "datasets to train algorithms that classify data or predict outcomes. "
        "Common algorithms include linear regression, logistic regression, and "
        "support vector machines (SVMs). The bias-variance trade-off is a central "
        "concept: models with high bias underfit data, while models with high "
        "variance overfit. Cross-validation helps estimate generalization error."
    ),
    "Lecture_02_Neural_Networks.pdf": (
        "Artificial neural networks are computing systems inspired by biological "
        "neural networks. A perceptron is the simplest form, computing a weighted "
        "sum of inputs passed through an activation function. Multi-layer "
        "perceptrons (MLPs) stack multiple layers—input, hidden, and output. "
        "Backpropagation computes gradients of the loss function with respect to "
        "each weight by the chain rule. Common activations: ReLU, sigmoid, tanh. "
        "Dropout randomly zeroes neurons during training to reduce overfitting."
    ),
    "Lecture_03_Deep_Learning.pdf": (
        "Deep learning extends neural networks with many hidden layers. "
        "Convolutional Neural Networks (CNNs) excel at image tasks using "
        "convolutional filters, pooling layers, and fully connected heads. "
        "Recurrent Neural Networks (RNNs) handle sequential data; LSTMs and GRUs "
        "address the vanishing gradient problem. Transfer learning allows reuse "
        "of pre-trained models (e.g., ResNet, VGG) on new tasks with fewer data."
    ),
    "Notes_Statistics_Refresher.md": (
        "# Statistics Refresher\n\n"
        "## Probability Distributions\n"
        "- **Normal distribution**: bell-shaped, defined by mean (μ) and std (σ).\n"
        "- **Bernoulli**: models binary outcomes.\n"
        "- **Poisson**: models count data over fixed intervals.\n\n"
        "## Hypothesis Testing\n"
        "- Null hypothesis H₀ vs alternative H₁.\n"
        "- p-value: probability of observing the data given H₀ is true.\n"
        "- Significance level α typically 0.05.\n"
        "- Type I error (false positive) vs Type II error (false negative)."
    ),
    "Exercises_Week1.txt": (
        "Exercise 1: Implement linear regression from scratch using NumPy.\n"
        "Exercise 2: Compare gradient descent vs closed-form solution.\n"
        "Exercise 3: Evaluate model using RMSE, MAE, and R² score.\n"
        "Exercise 4: Apply 5-fold cross-validation to a classification problem.\n"
        "Exercise 5: Visualize decision boundary of logistic regression on the "
        "Iris dataset."
    ),
}


# ── Canned response blocks ────────────────────────────────────────
# Each helper returns list[dict] in the standard response format.

def get_explanation_response(query=""):
    """Returns a rich explanation response."""
    return [
        {
            "category": "explanation",
            "content": {
                "title": "Understanding Backpropagation",
                "body": (
                    "**Backpropagation** is the cornerstone algorithm for training neural networks. "
                    "It works in two phases:\n\n"
                    "### 1. Forward Pass\n"
                    "Input data flows through the network layer by layer. Each neuron computes:\n\n"
                    "$$z = \\sum_{i} w_i x_i + b$$\n\n"
                    "The activation function (e.g., ReLU) is then applied: $a = \\sigma(z)$\n\n"
                    "### 2. Backward Pass\n"
                    "The **loss** (e.g., cross-entropy) is computed at the output. Gradients are "
                    "propagated backwards using the **chain rule**:\n\n"
                    "$$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial a} \\cdot "
                    "\\frac{\\partial a}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w}$$\n\n"
                    "### Key Takeaways\n"
                    "- Gradients tell us *how much* each weight contributed to the error.\n"
                    "- **Learning rate** controls the step size of weight updates.\n"
                    "- Vanishing gradients occur when activations (e.g., sigmoid) squash gradients "
                    "toward zero in deep networks — ReLU helps mitigate this."
                ),
            },
            "follow_ups": ["flashcard", "quiz", "flowchart"],
        }
    ]


def get_location_response(query=""):
    """Returns a corpus-location response."""
    return [
        {
            "category": "location",
            "content": {
                "file": "Lecture_02_Neural_Networks.pdf",
                "section": "Section 3 — Training",
                "page": "Page 12",
                "excerpt": (
                    "…Backpropagation computes gradients of the loss function with "
                    "respect to each weight by the chain rule. The gradient of the "
                    "loss is propagated from the output layer back through the "
                    "hidden layers…"
                ),
                "relevance": "Directly answers how neural network weights are updated during training.",
            },
            "follow_ups": ["explanation", "flashcard", "flowchart"],
        }
    ]


def get_flashcard_response(query=""):
    """Returns a set of flashcards."""
    return [
        {
            "category": "flashcard",
            "content": {
                "cards": [
                    {
                        "front": "What is the bias-variance trade-off?",
                        "back": (
                            "**Bias** measures error from wrong assumptions (underfitting). "
                            "**Variance** measures sensitivity to training data fluctuations "
                            "(overfitting). The goal is to find the sweet spot that minimizes "
                            "total error."
                        ),
                    },
                    {
                        "front": "What does the activation function do?",
                        "back": (
                            "It introduces **non-linearity** into the network, allowing it "
                            "to learn complex patterns. Without it, stacking layers would be "
                            "equivalent to a single linear transformation."
                        ),
                    },
                    {
                        "front": "What is dropout?",
                        "back": (
                            "A regularization technique that randomly sets a fraction of "
                            "neuron outputs to zero during training, forcing the network "
                            "to learn redundant representations and reducing overfitting."
                        ),
                    },
                    {
                        "front": "What is cross-validation?",
                        "back": (
                            "A technique to evaluate model performance by splitting data "
                            "into k folds, training on k-1 folds and testing on the "
                            "remaining fold, rotating through all folds."
                        ),
                    },
                    {
                        "front": "Name three common activation functions.",
                        "back": (
                            "**ReLU** — max(0, x), fast and avoids vanishing gradient.\n"
                            "**Sigmoid** — maps to (0,1), used for probabilities.\n"
                            "**Tanh** — maps to (-1,1), zero-centred output."
                        ),
                    },
                ],
            },
            "follow_ups": ["quiz", "explanation"],
        }
    ]


def get_quiz_response(query=""):
    """Returns a multiple-choice quiz."""
    return [
        {
            "category": "quiz",
            "content": {
                "title": "Neural Networks Quiz",
                "questions": [
                    {
                        "question": "Which algorithm is used to train neural networks by computing gradients?",
                        "options": [
                            "A) K-Nearest Neighbours",
                            "B) Backpropagation",
                            "C) Random Forest",
                            "D) Principal Component Analysis",
                        ],
                        "answer": "B",
                        "explanation": (
                            "Backpropagation uses the chain rule to compute gradients of "
                            "the loss w.r.t. each weight, enabling gradient descent updates."
                        ),
                    },
                    {
                        "question": "What problem do LSTMs solve compared to vanilla RNNs?",
                        "options": [
                            "A) Overfitting",
                            "B) Vanishing gradient problem",
                            "C) Data augmentation",
                            "D) Feature scaling",
                        ],
                        "answer": "B",
                        "explanation": (
                            "LSTMs use gating mechanisms (forget, input, output gates) to "
                            "control information flow, preventing gradients from vanishing "
                            "over long sequences."
                        ),
                    },
                    {
                        "question": "In a CNN, what does a pooling layer do?",
                        "options": [
                            "A) Increases spatial resolution",
                            "B) Applies non-linear activation",
                            "C) Reduces spatial dimensions while retaining key features",
                            "D) Normalises batch statistics",
                        ],
                        "answer": "C",
                        "explanation": (
                            "Pooling (e.g., max-pool) downsamples feature maps, reducing "
                            "computation and providing translation invariance."
                        ),
                    },
                ],
            },
            "follow_ups": ["flashcard", "explanation", "location"],
        }
    ]


def get_flowchart_response(query=""):
    """Returns a Mermaid flowchart."""
    return [
        {
            "category": "flowchart",
            "content": {
                "title": "Neural Network Training Pipeline",
                "mermaid": (
                    "graph TD\n"
                    "    A[Load Dataset] --> B[Split Train / Val / Test]\n"
                    "    B --> C[Initialize Weights]\n"
                    "    C --> D[Forward Pass]\n"
                    "    D --> E[Compute Loss]\n"
                    "    E --> F[Backward Pass - Backprop]\n"
                    "    F --> G[Update Weights - Optimizer]\n"
                    "    G --> H{Converged?}\n"
                    "    H -- No --> D\n"
                    "    H -- Yes --> I[Evaluate on Test Set]\n"
                    "    I --> J[Deploy Model]"
                ),
                "description": (
                    "This flowchart shows the standard deep learning training loop. "
                    "Data is split, weights initialised, and the model iterates through "
                    "forward passes, loss computation, and backpropagation until convergence."
                ),
            },
            "follow_ups": ["explanation", "quiz", "location"],
        }
    ]


# ── Master dispatcher ─────────────────────────────────────────────
RESPONSE_MAP = {
    "explanation": get_explanation_response,
    "location": get_location_response,
    "flashcard": get_flashcard_response,
    "quiz": get_quiz_response,
    "flowchart": get_flowchart_response,
}


def get_dummy_response(category, query=""):
    """Return a dummy response for the given category."""
    func = RESPONSE_MAP.get(category, get_explanation_response)
    return func(query)


def get_initial_dummy_response(query=""):
    """
    Simulate an initial LLM reply to a user query.
    Returns an explanation by default.
    """
    return get_explanation_response(query)

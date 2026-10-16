# Neural Language Modeling and Cross-Domain Autoregressive Token Generation

A production-structured, end-to-end PyTorch implementation of Neural Probabilistic Language Models for next-token prediction across English prose, C++ source code, and Python syntax. Developed as a final-year B.Tech capstone and technical placement showcase.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Theoretical Formulation](#theoretical-formulation)
3. [System Architecture](#system-architecture)
4. [Repository Structure](#repository-structure)
5. [Datasets and Tokenization Pipelines](#datasets-and-tokenization-pipelines)
6. [Trained Checkpoints and Model Specifications](#trained-checkpoints-and-model-specifications)
7. [Inference Engine and Decoding Strategies](#inference-engine-and-decoding-strategies)
8. [Embedding Space Analysis](#embedding-space-analysis)
9. [Installation and Environment Setup](#installation-and-environment-setup)
10. [Usage Guide](#usage-guide)
    - [Interactive Web Application](#interactive-web-application)
    - [Command-Line Generation](#command-line-generation)
    - [Model Evaluation and Latency Benchmarking](#model-evaluation-and-latency-benchmarking)
    - [Training Custom Corpora](#training-custom-corpora)
    - [Automated Unit Testing](#automated-unit-testing)
11. [Placement Interview Defense and Technical Q&A](#placement-interview-defense-and-technical-qa)
12. [References](#references)

---

## Executive Summary

Statistical language modeling is the task of assigning probability distributions over sequences of words or tokens. Traditional n-gram count models rely on Markovian assumptions and frequency tables, suffering acutely from the curse of dimensionality and zero-probability assignment for unseen sequences.

This repository implements the foundational Neural Probabilistic Language Model architecture proposed by Yoshua Bengio et al. (2003), extended across three diverse semantic domains:
- **Natural Language (English Prose)**: Trained on Leo Tolstoy's *War and Peace* (Project Gutenberg corpus), featuring a 19,764-word vocabulary.
- **Systems Programming (C++)**: Trained on the *llama.cpp* open-source codebase, featuring a 43,226-token lexical vocabulary covering C/C++ operators, structural delimiters, and identifiers.
- **High-Level Scripting (Python)**: Trained on the *GIANT Python* codebase, featuring a 521-token vocabulary for algorithmic scripting patterns.

The project features a modular Python package (`next_token_predictor`), configurable training and evaluation pipelines, reproducible inference scripts with temperature and top-k controls, comprehensive unit tests, and a Streamlit dashboard.

---

## Theoretical Formulation

### Autoregressive Language Modeling

Given an arbitrary sequence of discrete tokens $W = (w_1, w_2, \dots, w_T)$, the joint probability $P(W)$ is factored using the chain rule of conditional probability:

$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

Under an $n$-gram Markov assumption of order $N$, the conditional probability of token $w_t$ depends solely on the preceding $N-1$ context tokens:

$$P(w_t \mid w_1, \dots, w_{t-1}) \approx P(w_t \mid w_{t-N+1}, \dots, w_{t-1})$$

In this project, context window size $N$ is standardized to 5 tokens across models.

### Continuous Distributed Representations

Rather than representing words as sparse, mutually orthogonal one-hot vectors in $\mathbb{R}^{|V|}$, each discrete token index $w_i \in \{0, 1, \dots, |V|-1\}$ is projected into a continuous vector space $\mathbb{R}^D$ via an embedding matrix $E \in \mathbb{R}^{|V| \times D}$:

$$e_i = E(w_i) = E^T \mathbf{1}_{w_i}$$

Continuous embeddings map semantically and syntactically related words to proximate neighborhoods in the vector space, enabling the network to generalize to previously unseen n-gram combinations.

### Forward Propagation Equations

For an input context of $N$ previous tokens $c = (w_1, w_2, \dots, w_N)$:

1. **Embedding Lookup**:
   $$e_k = E(w_k) \in \mathbb{R}^D \quad \text{for } k \in \{1, \dots, N\}$$

2. **Feature Concatenation**:
   $$h_0 = \begin{bmatrix} e_1 \\ e_2 \\ \vdots \\ e_N \end{bmatrix} \in \mathbb{R}^{N \cdot D}$$

3. **Hidden Layer Projection**:
   $$h_1 = \sigma(W_1 h_0 + b_1) \in \mathbb{R}^H$$
   where $W_1 \in \mathbb{R}^{H \times (N \cdot D)}$, $b_1 \in \mathbb{R}^H$, and $\sigma(\cdot)$ represents a non-linear activation function (ReLU, Tanh, or Sigmoid).

4. **Vocabulary Logits Projection**:
   $$z = W_2 h_1 + b_2 \in \mathbb{R}^{|V|}$$
   where $W_2 \in \mathbb{R}^{|V| \times H}$ and $b_2 \in \mathbb{R}^{|V|}$.

5. **Softmax Output Distribution**:
   $$P(w_{t} = k \mid c) = \frac{\exp(z_k / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}$$
   where $T > 0$ represents the sampling temperature parameter.

### Loss Function and Optimization

The network parameters $\theta = \{E, W_1, b_1, W_2, b_2\}$ are optimized by minimizing the empirical Cross-Entropy Loss (Negative Log-Likelihood) over training mini-batches of size $B$:

$$\mathcal{L}(\theta) = -\frac{1}{B} \sum_{i=1}^{B} \log P(y_i \mid x_i; \theta)$$

Optimization is driven by the AdamW optimizer (Loshchilov & Hutter, 2019), which decouples weight decay regularization from gradient-based adaptive momentum updates:

$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

---

## System Architecture

```
                                  Input Prompt Context
                          ["the", "prince", "was", "silent", "and"]
                                             |
                                             v
                             +-------------------------------+
                             |    Domain-Specific Tokenizer  |
                             |   (TextTokenizer / CodeToken) |
                             +-------------------------------+
                                             |
                                   Token IDs: [14, 82, 9, 412, 6]
                                             |
                                             v
                             +-------------------------------+
                             |    Embedding Layer E          |
                             |    Shape: [|V|, D]            |
                             +-------------------------------+
                                             |
                                  Embeddings: [5 x D]
                                             |
                                             v
                             +-------------------------------+
                             |    Flatten / Concatenate      |
                             |    Shape: [1, 5 * D]          |
                             +-------------------------------+
                                             |
                                             v
                             +-------------------------------+
                             |    Linear Layer W1 + Bias b1  |
                             |    Shape: [5*D, H]            |
                             +-------------------------------+
                                             |
                                             v
                             +-------------------------------+
                             |    Non-Linear Activation      |
                             |    sigma in {ReLU, Tanh, Sig} |
                             +-------------------------------+
                                             |
                                             v
                             +-------------------------------+
                             |    Linear Layer W2 + Bias b2  |
                             |    Shape: [H, |V|]            |
                             +-------------------------------+
                                             |
                                   Logits z in R^|V|
                                             |
                                             v
                             +-------------------------------+
                             |    Temperature Scaling        |
                             |    z_scaled = z / T           |
                             +-------------------------------+
                                             |
                                             v
                             +-------------------------------+
                             |    Top-K Filtering (Optional) |
                             |    Keep top-k, mask to -inf   |
                             +-------------------------------+
                                             |
                                             v
                             +-------------------------------+
                             |    Softmax Probability Dist   |
                             |    p_i = exp(z_i) / sum(exp)  |
                             +-------------------------------+
                                             |
                                             v
                             +-------------------------------+
                             |    Multinomial / Greedy Sample|
                             |    Output: Next Token ID      |
                             +-------------------------------+
                                             |
                                             v
                                  Predicted Word / Token
                                         "looked"
```

---

## Repository Structure

```
NextTokenPredictor/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Continuous Integration workflow across Python 3.10-3.12
├── checkpoints/                       # Serialized PyTorch model checkpoints (.pt)
│   ├── war_and_peace/
│   │   ├── model1_task1.pt            # Hidden 1024, Embed 64, ReLU (Best English prose model)
│   │   ├── model2_task1.pt            # Hidden 128, Embed 32, Tanh
│   │   └── model3_task1.pt            # Hidden 128, Embed 64, Sigmoid
│   ├── cpp/
│   │   ├── cpp_next_token_model.pt    # Epoch 100 checkpoint
│   │   ├── cpp_next_token_model2.pt   # Epoch 300 checkpoint (Best C++ code model)
│   │   └── cpp_next_token_model3.pt   # Baseline checkpoint
│   └── python/
│       └── python_model1.pt           # Epoch 1000 Python script model
├── configs/                           # Declarative YAML experiment configurations
│   ├── war_and_peace.yaml             # Text corpus configuration
│   ├── cpp_code.yaml                  # C++ corpus configuration
│   └── python_code.yaml               # Python corpus configuration
├── data/
│   └── vocabularies/                  # Tokenizer mappings and vocabulary tables
│       ├── war_and_peace/
│       │   ├── stoi1.pkl              # String-to-Index dictionary (19,764 tokens)
│       │   ├── itos1.pkl              # Index-to-String dictionary (19,764 tokens)
│       │   └── stoi.pkl               # Canonical lookup dictionary
│       ├── cpp/
│       │   ├── cpp_vocabulary.csv     # Frequency distribution table (43,243 tokens)
│       │   ├── stoi_cpp.pkl           # Pre-compiled mapping (43,226 tokens)
│       │   └── itos_cpp.pkl           # Inverse mapping
│       └── python/
│           ├── python_vocabulary.csv  # Frequency table (521 tokens)
│           ├── stoi_python.pkl        # Pre-compiled mapping (521 tokens)
│           └── itos_python.pkl        # Inverse mapping
├── notebooks/                         # Interactive exploratory and training notebooks
│   ├── 01_natural_language_war_and_peace.ipynb  # Text preprocessing, training, and t-SNE analysis
│   ├── 02_code_modeling_cpp.ipynb               # C++ lexical extraction and training runs
│   └── 03_code_modeling_python.ipynb            # Python code modeling and visualization
├── src/                               # Modular core Python package
│   └── next_token_predictor/
│       ├── __init__.py                # Public API exports
│       ├── models/
│       │   ├── __init__.py
│       │   ├── mlp.py                 # Next_Word_Predictor, NextTokenPredictor, MLPNextTokenPredictor
│       │   └── lstm.py                # Recurrent LSTM comparative baseline
│       ├── tokenization/
│       │   ├── __init__.py
│       │   ├── text_tokenizer.py      # Natural language word-level tokenizer
│       │   └── code_tokenizer.py      # Regex-based lexical code tokenizer
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── loader.py              # Robust PyTorch checkpoint and vocabulary loader
│       │   └── generator.py           # Autoregressive generation engine with temperature and top-k
│       ├── training/
│       │   ├── __init__.py
│       │   ├── dataset.py             # Sliding-window PyTorch Dataset builder
│       │   └── trainer.py             # Standardized model training manager
│       └── utils/
│           ├── __init__.py
│           ├── device.py              # Hardware accelerator detection (CUDA, MPS, CPU)
│           ├── metrics.py             # Perplexity and accuracy calculations
│           └── visualization.py       # Embedding projection via t-SNE and PCA
├── scripts/                           # Standalone CLI utilities
│   ├── train.py                       # Command-line training runner
│   ├── evaluate.py                    # Model capacity, perplexity, and latency benchmarker
│   └── generate.py                    # Command-line text and code generation runner
├── tests/                             # Automated unit testing suite
│   ├── __init__.py
│   ├── test_models.py                 # Forward pass and parameter tests
│   ├── test_tokenizers.py             # Normalization and lexing tests
│   └── test_inference.py              # Autoregressive loop and checkpoint loading tests
├── web/                               # Streamlit web application
│   └── app.py                         # Multi-domain interactive generation dashboard
├── app.py                             # Root wrapper for Streamlit Cloud deployment
├── app2.py                            # Backward-compatible entry point
├── model_utils.py                     # Legacy compatibility shim
├── model_utils2.py                    # Legacy compatibility shim
├── pyproject.toml                     # Modern PEP 517/518 build metadata and pytest configuration
├── setup.py                           # Setuptools installation configuration
├── requirements.txt                   # Production pinned dependencies
├── runtime.txt                        # Target Python runtime environment
└── README.md                          # Project documentation
```

---

## Datasets and Tokenization Pipelines

### 1. Natural Language Prose (Leo Tolstoy's War and Peace)
- **Source**: Project Gutenberg / Kaggle plain-text release.
- **Preprocessing**:
  1. Hyphen normalization: `-` replaced with single spaces.
  2. Character stripping: non-alphanumeric characters eliminated except for periods (`.`).
  3. Case folding: full text lowercased.
  4. Sentence delimiter isolation: periods padded with whitespace to function as distinct boundary tokens (`.` $\to$ ` . `).
- **Vocabulary Size**: 19,764 unique words.
- **Out-of-Vocabulary (OOV) Strategy**: Unseen words map to token index 0 (`<unk>`).

### 2. C++ Source Code (LLaMA.cpp)
- **Source**: The open-source `llama.cpp` codebase (GGML engine).
- **Lexical Pattern**: Tokenized using regular expressions capturing C/C++ identifiers, keywords, types, arithmetic/logical operators, and structural syntax delimiters:
  ```python
  re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[{}()\[\];.,=+\-*/<>!&|^%~?:]")
  ```
- **Vocabulary Size**: 43,226 tokens.
- **Data Engineering Caveat**: In standard CSV parsers (such as `pandas.read_csv`), the string token `'NULL'` is parsed as `NaN` by default. The pipeline specifies `keep_default_na=False` to preserve `'NULL'` as an active C++ pointer literal.

### 3. Python Source Code (GIANT Repository)
- **Source**: Deep learning scripting routines from the `GIANT-Python-Code` repository.
- **Lexical Pattern**: Identical regex tokenizer targeting Python keywords (`def`, `self`, `import`, `return`), numerical constants, and indentation-related delimiters.
- **Vocabulary Size**: 521 unique tokens.

---

## Trained Checkpoints and Model Specifications

| Domain | Model Checkpoint | Parameters | Vocab Size ($|V|$) | Context ($N$) | Embed Dim ($D$) | Hidden Dim ($H$) | Activation | Epochs | Checkpoint Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Text** | Model 1 | 21,851,700 | 19,764 | 5 | 64 | 1024 | ReLU | 100 | `checkpoints/war_and_peace/model1_task1.pt` |
| **Text** | Model 2 | 3,212,508 | 19,764 | 5 | 32 | 128 | Tanh | 100 | `checkpoints/war_and_peace/model2_task1.pt` |
| **Text** | Model 3 | 3,845,020 | 19,764 | 5 | 64 | 128 | Sigmoid | 100 | `checkpoints/war_and_peace/model3_task1.pt` |
| **C++** | Model 2 | 13,911,450 | 43,226 | 5 | 64 | 256 | ReLU | 300 | `checkpoints/cpp/cpp_next_token_model2.pt` |
| **C++** | Model 1 | 13,911,450 | 43,226 | 5 | 64 | 256 | ReLU | 100 | `checkpoints/cpp/cpp_next_token_model.pt` |
| **C++** | Model 3 | 6,957,018 | 43,226 | 5 | 32 | 128 | ReLU | 100 | `checkpoints/cpp/cpp_next_token_model3.pt` |
| **Python** | Model 1 | 180,441 | 521 | 5 | 64 | 256 | ReLU | 1000 | `checkpoints/python/python_model1.pt` |

### Architectural Insights & Ablations
1. **Capacity vs Generalization**: Model 1 (1024 hidden units) captures nuanced syntactic constructs and multi-word semantic bindings significantly better than Model 2 (128 units), at the cost of a higher memory footprint (83.36 MB vs 12.25 MB).
2. **Activation Dynamics**: ReLU non-linearities avoided the vanishing gradient saturation observed during long training runs with Sigmoid (Model 3), resulting in faster convergence and lower terminal cross-entropy loss.
3. **Parameter Distribution**: In all models, the output projection matrix $W_2 \in \mathbb{R}^{|V| \times H}$ accounts for over 85% of total trainable parameters, demonstrating the parameter concentration inherent to wide vocabulary language modeling (the Softmax Bottleneck).

---

## Inference Engine and Decoding Strategies

Autoregressive generation generates tokens sequentially by appending the predicted token at step $t$ to the context window and shifting the sliding window by one position to predict step $t+1$.

### Temperature Scaling

The temperature parameter $T \in (0, \infty)$ re-scales the logit landscape prior to exponentiation:

$$p_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$

- **Low Temperature ($T < 0.7$)**: Exaggerates probability differentials. As $T \to 0$, the distribution concentrates on $\operatorname{argmax}(z)$, yielding conservative, highly repetitive text.
- **Balanced Temperature ($T \in [0.7, 1.0]$)**: Balances diversity with coherence.
- **High Temperature ($T > 1.2$)**: Flattens the probability distribution toward uniformity, increasing vocabulary diversity at the expense of syntactic fidelity.

### Top-K Truncation Filtering

To prevent sampling from the long tail of low-probability, grammatically incorrect tokens, Top-K filtering restricts the candidate set to the $K$ highest logits:

$$V^{(K)} = \operatorname{argtopk}_{k}(z)$$
$$z_i' = \begin{cases} z_i & \text{if } i \in V^{(K)} \\ -\infty & \text{otherwise} \end{cases}$$

Softmax is then evaluated over $z'$, ensuring zero probability mass is allocated to tokens outside the top $K$.

---

## Embedding Space Analysis

The learned embedding matrix $E \in \mathbb{R}^{|V| \times D}$ maps words into continuous geometric vector space. In `notebooks/01_natural_language_war_and_peace.ipynb`, t-Distributed Stochastic Neighbor Embedding (t-SNE) and Principal Component Analysis (PCA) are applied to extract 2D projections of token subsets:

1. **Syntactic Parts of Speech**: Pronouns (`i`, `you`, `he`, `she`, `we`, `they`, `me`, `him`, `her`) form a tight geometric cluster isolated from verbs and nouns, demonstrating that the embedding layer captures grammatical equivalence without explicit supervision.
2. **Structural Punctuation**: Sentence boundary markers (`.`) locate away from descriptive lexical terms, reflecting their distinct syntactic function in sliding context windows.
3. **Source Code Primitives**: In C++ embeddings, binary operators (`+`, `-`, `*`, `/`) cluster together, while structural delimiters (`{`, `}`, `(`, `)`) form distinct functional neighborhoods.

---

## Installation and Environment Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/SiddheshP2709/NextTokenPredictor.git
cd NextTokenPredictor
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies and Package in Editable Mode
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Usage Guide

### Interactive Web Application

Launch the multi-domain Streamlit dashboard:

```bash
streamlit run web/app.py
```

The application provides:
- Seamless switching between Natural Language, C++, and Python domains.
- Real-time adjustment of Temperature and Top-K parameters.
- Token probability distribution tables and bar charts for top candidates at each step.
- Interactive technical architecture documentation and viva interview guide.

### Command-Line Generation

Generate text and code completions directly from the command line:

```bash
# 1. Natural Language (War & Peace)
python scripts/generate.py \
    --domain text \
    --prompt "the prince was silent and" \
    --tokens 10 \
    --temperature 0.8 \
    --track-probs

# 2. C++ Source Code (llama.cpp)
python scripts/generate.py \
    --model-path checkpoints/cpp/cpp_next_token_model2.pt \
    --vocab-path data/vocabularies/cpp/stoi_cpp.pkl \
    --domain cpp \
    --prompt "int calculate_sum(int count)" \
    --tokens 12 \
    --temperature 0.7

# 3. Python Source Code (GIANT)
python scripts/generate.py \
    --model-path checkpoints/python/python_model1.pt \
    --vocab-path data/vocabularies/python/stoi_python.pkl \
    --domain python \
    --prompt "def calculate_loss(self, x):" \
    --tokens 10 \
    --temperature 0.6
```

### Model Evaluation and Latency Benchmarking

Evaluate model capacity, parameter count, memory footprint, and forward-pass inference latency:

```bash
python scripts/evaluate.py \
    --model-path checkpoints/war_and_peace/model1_task1.pt \
    --vocab-path data/vocabularies/war_and_peace/stoi1.pkl
```

Sample Benchmark Output:
```
==================================================
MODEL ARCHITECTURE & CAPACITY
==================================================
Model Class:           Next_Word_Predictor
Device:                cpu
Context Window Size:   5
Vocabulary Size:       19,764
Total Parameters:      21,851,700
Trainable Parameters:  21,851,700
Model Memory Size:     83.36 MB
==================================================

==================================================
INFERENCE LATENCY BENCHMARK
==================================================
Mean Forward Latency:  1.828 ms per step
Throughput:            546.9 tokens/sec
==================================================
```

### Training Custom Corpora

Train a new language model from scratch on arbitrary text or code:

```bash
python scripts/train.py \
    --architecture mlp \
    --context-size 5 \
    --embed-dim 64 \
    --hidden-dim 256 \
    --batch-size 1024 \
    --epochs 20 \
    --lr 0.001 \
    --output-dir checkpoints/custom
```

### Automated Unit Testing

Run the test suite via pytest:

```bash
pytest -v
```

All 10 unit tests cover forward-pass dimensionality, tokenizer edge cases, probability tracking, and checkpoint loading.

---

## Placement Interview Defense and Technical Q&A

This section provides rigorous technical answers to common questions asked by interviewers evaluating machine learning and language modeling projects.

### 1. What motivated the transition from n-gram count models to neural language models?
Traditional n-gram count models (such as trigram or 5-gram models with Kneser-Ney smoothing) store probability tables estimated via frequency ratios:

$$P(w_t \mid w_{t-1}, w_{t-2}) = \frac{\text{Count}(w_{t-2}, w_{t-1}, w_t)}{\text{Count}(w_{t-2}, w_{t-1})}$$

This approach suffers from two severe limitations:
1. **Curse of Dimensionality and Sparsity**: As context length $N$ increases, the table size grows as $O(|V|^N)$. For vocabulary size $|V| = 20,000$ and $N = 5$, the state space exceeds $3.2 \times 10^{21}$, making empirical frequency estimation impossible.
2. **Lack of Semantic Generalization**: If the training set contains "the cat sat on the mat", a count model learns nothing about "the dog sat on the rug" because word tokens are discrete, independent symbols. Neural language models solve this by mapping words to continuous embedding vectors where semantic similarity is captured geometrically.

### 2. What are the key architectural differences between an n-gram MLP, an LSTM, and a Transformer?

| Dimension | N-gram MLP (Bengio 2003) | Recurrent LM (LSTM) | Transformer (Vaswani 2017 / GPT) |
| :--- | :--- | :--- | :--- |
| **Context Length** | Fixed ($N$ tokens) | Theoretically infinite, practically limited (~100-200) | Long ($2,048$ to $128,000+$ tokens) |
| **Path Length Between Tokens** | $O(1)$ through concatenated projection | $O(T)$ sequential recurrence steps | $O(1)$ direct pairwise self-attention |
| **Temporal Invariance** | Order encoded via concatenation indices | Natural sequential state transition ($h_t = f(h_{t-1}, x_t)$) | Requires explicit Positional Embeddings |
| **Training Parallelization** | Fully parallelizable across mini-batch samples | Non-parallelizable across sequence length ($O(T)$ sequential dependencies) | Fully parallelizable across both batch and sequence ($O(1)$ depth) |
| **Computational Bottleneck** | Projection $W_1$ and Softmax $W_2$ | Recurrent gate matrix multiplications | Self-attention quadratic complexity $O(T^2)$ |

### 3. What is the "Softmax Bottleneck" and how does it manifest in your model?
The Softmax Bottleneck refers to the representational limitation imposed by the final linear projection layer:

$$z = W_2 h_1 + b_2$$

Because $W_2 \in \mathbb{R}^{|V| \times H}$ has rank bounded by $\min(|V|, H) = H$, and typically $H \ll |V|$ (e.g., $H=1024$ while $|V|=19,764$), the matrix of log-probabilities for all possible contexts has rank at most $H$. In our Natural Language Model 1, the output layer requires $1024 \times 19,764 = 20,238,336$ parameters, representing **92.6% of the total model parameters**. The hidden layer represents an information bottleneck through which all linguistic distribution modes must be expressed.

### 4. How does Temperature Scaling mathematically modify the Softmax distribution?
The Softmax distribution with temperature $T$ is given by:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Taking the partial derivative with respect to logit $z_i$:

$$\frac{\partial P(w_i)}{\partial z_i} = \frac{1}{T} P(w_i) (1 - P(w_i))$$

- As $T \to 0^+$, for the maximal logit $z^* = \max_j z_j$:
  $$\lim_{T \to 0^+} \frac{\exp((z^* - z_i)/T)}{\sum_j \exp((z_j - z_i)/T)} = \begin{cases} 1 & \text{if } z_i = z^* \\ 0 & \text{otherwise} \end{cases}$$
  The distribution degenerates into a Kronecker delta function (equivalent to greedy selection).
- As $T \to \infty$, $z_i / T \to 0$ for all $i$:
  $$\lim_{T \to \infty} P(w_i) = \frac{\exp(0)}{\sum_{j=1}^{|V|} \exp(0)} = \frac{1}{|V|}$$
  The distribution converges to a uniform distribution across the entire vocabulary.

### 5. Why is Top-K sampling preferred over pure Multinomial sampling or Greedy Decoding?
- **Greedy Decoding ($\operatorname{argmax}$)**: Frequently causes degenerative loops (repetitive cycles of phrases) because language is inherently multimodal; the globally optimal sequence is not necessarily the concatenation of locally greedy steps.
- **Pure Multinomial Sampling**: Samples from the entire distribution, including the long tail of tens of thousands of tokens with probabilities on the order of $10^{-5}$. Over a multi-token generation sequence, sampling an inappropriate tail token cascades errors and leads to grammatical collapse.
- **Top-K Sampling**: Truncates the tail by taking the top $K$ candidate tokens, renormalizing their probabilities to sum to 1. This retains lexical diversity while strictly bounding generation within the high-confidence probability mass.

### 6. What is Perplexity (PPL) and how does it relate to Cross-Entropy?
Perplexity is the exponentiated average negative log-likelihood per token:

$$\text{PPL} = \exp(\mathcal{L}) = \exp\left( -\frac{1}{T} \sum_{t=1}^T \log P(w_t \mid w_{<t}) \right)$$

Information-theoretically, Perplexity equals $2^{H(P, Q)}$ when using base-2 logarithms, representing the cross-entropy between the empirical data distribution $P$ and model distribution $Q$. Intuition: a Perplexity of $K$ indicates that at each step, the language model is as uncertain as if it were choosing uniformly among $K$ equally likely candidate tokens. A lower perplexity indicates superior predictive confidence.

### 7. How does tokenization differ between natural language and source code?
Natural language tokenization targets whitespace and orthographic punctuation, relying on vocabulary frequencies where inflected forms (`walk`, `walks`, `walking`) map to distinct indices.

In contrast, source code tokenization requires lexical analysis:
1. **Identifiers vs Keywords**: Distinguishing language syntax primitives (`int`, `for`, `class`, `def`) from arbitrary user identifiers (`total_sum`, `llama_context`).
2. **Operator Disambiguation**: Multi-character operators (`==`, `!=`, `->`, `<<`) must be parsed as distinct atomic tokens rather than individual punctuation marks.
3. **Punctuation Significance**: In natural language, a missing period is syntactically tolerable; in C++, a missing semicolon (`;`) or unclosed brace (`}`) renders the sequence syntactically invalid. The lexical tokenizer in `code_tokenizer.py` isolates structural delimiters to allow the model to learn grammatical transitions.

### 8. Explain the data engineering issue encountered with C++ vocabulary extraction and how you resolved it.
When exporting token frequency counts to CSV and reloading via `pandas.read_csv`, the C/C++ literal token `'NULL'` is interpreted as a missing value (`NaN`) by default, alongside other string literals like `'NA'` or `'nan'`. If unhandled, this produces null entries in the vocabulary table, misaligns integer-to-string dictionary indices, and causes runtime `KeyError` exceptions during generation.

We resolved this by strictly specifying `keep_default_na=False` in `CodeTokenizer.from_csv` and serializing verified dictionary objects directly to binary pickle files (`stoi_cpp.pkl`, `itos_cpp.pkl`), ensuring that literal programming constants remain intact.

### 9. Why did you choose AdamW over standard Stochastic Gradient Descent (SGD)?
Standard SGD with weight decay adds an $L_2$ regularization penalty to the objective function:

$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|_2^2$$

In standard Adam, the gradient of the penalty term is combined with the objective gradient inside the first and second moment estimates ($m_t$ and $v_t$). Consequently, parameters with large historical gradients receive smaller effective weight decay, whereas parameters with infrequent updates receive disproportionately large decay. AdamW decouples weight decay from the gradient moments, applying weight decay directly to the parameter update:

$$\theta_{t+1} = \theta_t - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta_t \lambda \theta_t$$

This improves generalization across sparse embedding layers where token update frequencies vary by orders of magnitude.

### 10. If deploying this model in a low-latency production environment, what optimizations would you apply?
1. **Post-Training Quantization (PTQ)**: Quantize model weights from FP32 (32-bit float) to INT8 (8-bit integer), reducing memory footprint by 75% (e.g., from 83 MB to ~21 MB) and utilizing vectorized INT8 matrix instructions (such as AVX-512 VNNI) for 2x to 4x latency improvements.
2. **ONNX Runtime Export**: Export the PyTorch model graph to an optimized ONNX computational graph, fusing linear layers with activations (`Linear` + `ReLU` fusion) and eliminating Python interpreter overhead.
3. **Embedding Table Pruning**: Frequency-based vocabulary pruning, truncating tail tokens that occur fewer than $M$ times and reallocating their mass to `<unk>`, compressing the output layer $W_2$ substantially.

---

## References

1. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model*. Journal of Machine Learning Research, 3, 1137-1155.
2. Karpathy, A. (2022). *makemore: An autoregressive character-level language model*. GitHub repository.
3. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization (AdamW)*. International Conference on Learning Representations (ICLR).
4. Vaswani, A., et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 2017).
5. Gerganov, G., et al. (2023). *llama.cpp: Port of Facebook's LLaMA model in C/C++*. GitHub repository.

"""Interactive Streamlit Demonstration Application for Next Token Prediction.

Provides multi-domain sequence generation, real-time probability inspection,
and model architecture exploration across Natural Language, C++, and Python domains.
"""

import sys
import os
import re
import pandas as pd
import streamlit as st
import torch

# Ensure package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from next_token_predictor.inference.loader import load_checkpoint, load_vocabulary, resolve_path
from next_token_predictor.inference.generator import TextGenerator
from next_token_predictor.utils.device import get_device


# Page Configuration
st.set_page_config(
    page_title="Neural Language Model - Next Token Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Professional Placement Presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .token-chip {
        display: inline-block;
        background-color: #EEF2F6;
        color: #0F172A;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .stCodeBlock {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# Domain Registry
DOMAINS = {
    "Natural Language (War & Peace)": {
        "domain_key": "text",
        "vocab_path": "data/vocabularies/war_and_peace/stoi1.pkl",
        "itos_path": "data/vocabularies/war_and_peace/itos1.pkl",
        "models": {
            "Model 1 (Hidden 1024, Embed 64, ReLU)": {
                "path": "checkpoints/war_and_peace/model1_task1.pt",
                "hidden_dim": 1024,
                "embed_dim": 64,
                "activation": "ReLU",
                "epochs": 100,
            },
            "Model 2 (Hidden 128, Embed 32, Tanh)": {
                "path": "checkpoints/war_and_peace/model2_task1.pt",
                "hidden_dim": 128,
                "embed_dim": 32,
                "activation": "Tanh",
                "epochs": 100,
            },
            "Model 3 (Hidden 128, Embed 64, Sigmoid)": {
                "path": "checkpoints/war_and_peace/model3_task1.pt",
                "hidden_dim": 128,
                "embed_dim": 64,
                "activation": "Sigmoid",
                "epochs": 100,
            },
        },
        "presets": [
            "the prince was silent and",
            "all the guests were",
            "pierre looked at him with",
            "in the middle of the room",
        ],
    },
    "C++ Source Code (LLaMA.cpp)": {
        "domain_key": "cpp",
        "vocab_path": "data/vocabularies/cpp/stoi_cpp.pkl",
        "itos_path": "data/vocabularies/cpp/itos_cpp.pkl",
        "models": {
            "Checkpoint 2 (Epoch 300, Hidden 256, Embed 64)": {
                "path": "checkpoints/cpp/cpp_next_token_model2.pt",
                "hidden_dim": 256,
                "embed_dim": 64,
                "activation": "ReLU",
                "epochs": 300,
            },
            "Checkpoint 1 (Epoch 100, Hidden 256, Embed 64)": {
                "path": "checkpoints/cpp/cpp_next_token_model.pt",
                "hidden_dim": 256,
                "embed_dim": 64,
                "activation": "ReLU",
                "epochs": 100,
            },
            "Checkpoint 3 (Epoch 100 Baseline, Hidden 128, Embed 32)": {
                "path": "checkpoints/cpp/cpp_next_token_model3.pt",
                "hidden_dim": 128,
                "embed_dim": 32,
                "activation": "ReLU",
                "epochs": 100,
            },
        },
        "presets": [
            "int calculate_sum(int count)",
            "struct llama_context {",
            "static void ggml_graph_compute(",
            "if (token == NULL)",
        ],
    },
    "Python Source Code (GIANT Repository)": {
        "domain_key": "python",
        "vocab_path": "data/vocabularies/python/stoi_python.pkl",
        "itos_path": "data/vocabularies/python/itos_python.pkl",
        "models": {
            "Checkpoint 1 (Epoch 1000, Hidden 256, Embed 64)": {
                "path": "checkpoints/python/python_model1.pt",
                "hidden_dim": 256,
                "embed_dim": 64,
                "activation": "ReLU",
                "epochs": 1000,
            },
        },
        "presets": [
            "def calculate_loss(self, x):",
            "import numpy as np",
            "for i in range(len(",
            "plt . semilogy (",
        ],
    },
}


@st.cache_resource(show_spinner=False)
def get_cached_resources(model_path: str, vocab_path: str, itos_path: str):
    """Load model and vocabulary with persistent caching."""
    model, device = load_checkpoint(model_path)
    stoi, itos = load_vocabulary(vocab_path, itos_path)
    return model, stoi, itos, device


# Header
st.markdown('<div class="main-title">Neural Language Model: Next Token Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Autoregressive sequence continuation using n-gram neural probabilistic architectures across English Prose, C++, and Python domains.</div>',
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.header("Configuration & Hyperparameters")

# Domain Selection
domain_name = st.sidebar.selectbox("Corpus Domain:", list(DOMAINS.keys()))
domain_info = DOMAINS[domain_name]

# Model Variant Selection
model_choice = st.sidebar.selectbox("Model Architecture / Variant:", list(domain_info["models"].keys()))
selected_model_meta = domain_info["models"][model_choice]

# Sampling Hyperparameters
st.sidebar.subheader("Sampling Hyperparameters")
temperature = st.sidebar.slider(
    "Temperature (Softmax Scaling):",
    min_value=0.1,
    max_value=2.0,
    value=0.8,
    step=0.05,
    help="Higher values increase entropy and diversity; lower values collapse towards greedy decoding.",
)

enable_top_k = st.sidebar.checkbox("Enable Top-K Filtering", value=False)
top_k_val = None
if enable_top_k:
    top_k_val = st.sidebar.slider("Top-K Candidates:", min_value=1, max_value=50, value=15)

max_tokens = st.sidebar.slider("Tokens to Predict:", min_value=1, max_value=50, value=10)
seed_val = st.sidebar.number_input("Random Seed (Reproducibility):", min_value=1, max_value=999999, value=42)

# Load Selected Artifacts
with st.spinner("Loading checkpoint and vocabulary..."):
    model, stoi, itos, device = get_cached_resources(
        selected_model_meta["path"],
        domain_info["vocab_path"],
        domain_info["itos_path"],
    )

context_size = getattr(model, "size", getattr(model, "context_size", 5))
vocab_size = len(stoi)
total_params = sum(p.numel() for p in model.parameters())

# Sidebar Metadata
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Model Metadata**
- Device: `{device}`
- Vocabulary Size: `{vocab_size:,}` tokens
- Context Window ($N$): `{context_size}` tokens
- Parameters: `{total_params:,}`
- Hidden Units: `{selected_model_meta['hidden_dim']}`
- Embedding Dimension: `{selected_model_meta['embed_dim']}`
- Activation: `{selected_model_meta['activation']}`
""")

# Main Tabs
tab_inference, tab_theory, tab_viva = st.tabs([
    "Interactive Generation",
    "Architecture & Mathematics",
    "Interview & Defense Guide",
])

with tab_inference:
    col_input, col_meta = st.columns([2.5, 1])

    with col_input:
        st.subheader("Prompt Input")
        preset_choice = st.selectbox(
            "Select Prompt Preset:",
            ["(Custom Prompt)"] + domain_info["presets"],
        )

        default_text = preset_choice if preset_choice != "(Custom Prompt)" else domain_info["presets"][0]
        prompt = st.text_input("Seed Context:", value=default_text)

        generate_clicked = st.button("Generate Next Tokens", type="primary")

    with col_meta:
        st.subheader("System Status")
        st.metric(label="Vocabulary Size", value=f"{vocab_size:,}")
        st.metric(label="Parameter Count", value=f"{total_params:,}")
        st.metric(label="Context Size", value=f"{context_size} tokens")

    if generate_clicked:
        if not prompt.strip():
            st.error("Please enter a non-empty starting prompt.")
        else:
            generator = TextGenerator(
                model=model,
                stoi=stoi,
                itos=itos,
                device=device,
                context_size=context_size,
                domain=domain_info["domain_key"],
            )

            with st.spinner("Running autoregressive forward pass..."):
                result = generator.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k_val,
                    seed=seed_val,
                    track_probabilities=True,
                )

            st.markdown("---")
            st.subheader("Generation Output")

            if domain_info["domain_key"] in ("cpp", "python"):
                st.code(result["full_text"], language=domain_info["domain_key"])
            else:
                st.write(result["full_text"])

            st.markdown("##### Generated Token Sequence")
            chips_html = "".join([f'<span class="token-chip">{tok}</span>' for tok in result["generated_tokens"]])
            st.markdown(chips_html, unsafe_allow_html=True)

            # Step-by-Step Probability Inspection
            if result["steps"]:
                st.markdown("---")
                st.subheader("Token Probability Distribution (Top Candidates per Step)")
                st.caption("Visualizing the categorical distribution output by Softmax(logits / T).")

                prob_rows = []
                for step in result["steps"]:
                    row = {"Step": f"Step {step['step']}"}
                    for rank, (cand, prob) in enumerate(step["candidates"][:3], start=1):
                        row[f"Rank {rank} Token"] = cand
                        row[f"Rank {rank} Prob"] = f"{prob * 100:.1f}%"
                    prob_rows.append(row)

                df_probs = pd.DataFrame(prob_rows)
                st.dataframe(df_probs, use_container_width=True)

                # Chart for Step 1
                first_step = result["steps"][0]
                chart_data = pd.DataFrame({
                    "Token": [c[0] for c in first_step["candidates"]],
                    "Probability": [c[1] for c in first_step["candidates"]],
                }).set_index("Token")
                st.bar_chart(chart_data)

with tab_theory:
    st.subheader("Theoretical Formulation: Neural Probabilistic Language Model")
    st.markdown(r"""
    This project implements an n-gram Neural Probabilistic Language Model inspired by Yoshua Bengio et al. (2003)
    and Andrej Karpathy's foundational architectures.
    
    #### 1. Language Modeling Objective
    Given a sequence of tokens $w_1, w_2, \dots, w_T$, the joint probability is factored via the chain rule of probability:
    $$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \\mid w_1, \dots, w_{t-1})$$
    
    Under the Markov assumption with context window size $N$:
    $$P(w_t \\mid w_1, \dots, w_{t-1}) \\approx P(w_t \\mid w_{t-N+1}, \dots, w_{t-1})$$

    #### 2. Model Forward Computation
    1. **Embedding Layer**: Each discrete token index $w_i \\in \\{0, \\dots, |V|-1\\}$ is projected into continuous distributed vector space:
       $$e_i = E(w_i) \\in \\mathbb{R}^{D}$$
    2. **Concatenation**: Context embeddings are concatenated:
       $$h_0 = [e_1; e_2; \\dots; e_N] \\in \\mathbb{R}^{N \\times D}$$
    3. **Hidden Representation**:
       $$h_1 = \\sigma(W_1 \\cdot h_0 + b_1) \\in \\mathbb{R}^{H}$$
       Where $\\sigma$ is non-linear activation (ReLU, Tanh, or Sigmoid).
    4. **Unnormalized Logits Output**:
       $$z = W_2 \\cdot h_1 + b_2 \\in \\mathbb{R}^{|V|}$$
    5. **Softmax Normalization & Temperature Scaling**:
       $$P(w_t = k) = \\frac{\\exp(z_k / T)}{\\sum_{j=1}^{|V|} \\exp(z_j / T)}$$

    #### 3. Loss Function
    Trained end-to-end via Cross-Entropy Loss (Negative Log-Likelihood):
    $$\\mathcal{L} = - \\sum_{k=1}^{|V|} y_k \\log(P(w_t = k)) = -\\log P(w_t = y^*)$$
    """)

with tab_viva:
    st.subheader("Placement Interview Reference & Technical Viva Q&A")
    st.markdown(r"""
    #### Key Questions Interviewers Will Ask About This Project:

    **Q1: Why use learned embeddings instead of one-hot encoding?**
    - *Answer:* One-hot vectors are orthogonal and equidistant; they cannot capture semantic or syntactic similarity (cosine similarity is always 0). Furthermore, one-hot vectors scale with $O(|V|)$, causing dimensionality explosion. Learned dense embeddings $E \\in \\mathbb{R}^{|V| \\times D}$ project words into a continuous geometric manifold where semantically related tokens cluster together.

    **Q2: What is the primary limitation of this fixed-window MLP architecture compared to Transformers or LSTMs?**
    - *Answer:* The MLP architecture enforces a rigid, fixed-length context window $N$. Any context beyond $N$ tokens is entirely lost. In contrast, Recurrent Neural Networks (RNN/LSTM) maintain a hidden state vector over time, and Transformers utilize multi-head self-attention with dynamic $O(1)$ path length between any two tokens in sequence length $L$.

    **Q3: How does sampling temperature $T$ mathematically affect generation?**
    - *Answer:* As $T \\to 0$, the scaled logits $z/T$ amplify the gap between the largest logit and all others, collapsing the softmax distribution into a Dirac delta function (equivalent to greedy argmax). As $T \\to \\infty$, $z/T \\to 0$, causing softmax to converge toward a uniform distribution across all vocabulary tokens.

    **Q4: Why was AdamW chosen over standard SGD?**
    - *Answer:* AdamW decouples $L_2$ weight decay regularization from gradient-based adaptive moment estimation, preventing weight decay distortion in parameters with large gradient variance.
    """)

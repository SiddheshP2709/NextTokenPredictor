"""Interactive Streamlit Application for Next Word Prediction.

A clean, undergraduate-level course project demonstrating next word prediction
using Neural Language Models (MLP) trained on Leo Tolstoy's War and Peace.
"""

import sys
import os
import re
import pandas as pd
import streamlit as st
import torch

# Ensure package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from next_token_predictor.inference.loader import load_checkpoint, load_vocabulary
from next_token_predictor.inference.generator import TextGenerator


# Page Configuration
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Clean Academic Presentation
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
    .prompt-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px;
        font-size: 1.15rem;
        line-height: 1.6;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .generated-word {
        color: #2563EB;
        font-weight: 600;
        background-color: #DBEAFE;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 0 2px;
    }
    .metric-badge {
        display: inline-block;
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 6px 12px;
        margin-right: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Models Registry (War and Peace Models)
MODELS = {
    "Model 1 (ReLU - 1024 Hidden Units)": {
        "path": "checkpoints/war_and_peace/model1_task1.pt",
        "hidden_dim": 1024,
        "embed_dim": 64,
        "activation": "ReLU",
        "desc": "Largest model with ReLU activation. Captures rich vocabulary patterns.",
    },
    "Model 2 (Tanh - 128 Hidden Units)": {
        "path": "checkpoints/war_and_peace/model2_task1.pt",
        "hidden_dim": 128,
        "embed_dim": 32,
        "activation": "Tanh",
        "desc": "Compact model with hyperbolic tangent (Tanh) activation.",
    },
    "Model 3 (Sigmoid - 128 Hidden Units)": {
        "path": "checkpoints/war_and_peace/model3_task1.pt",
        "hidden_dim": 128,
        "embed_dim": 64,
        "activation": "Sigmoid",
        "desc": "Baseline model using Sigmoid activation.",
    },
}

VOCAB_PATH = "data/vocabularies/war_and_peace/stoi1.pkl"
ITOS_PATH = "data/vocabularies/war_and_peace/itos1.pkl"

SAMPLE_PROMPTS = [
    "the prince was silent and",
    "in the middle of the",
    "all the guests were",
    "pierre looked at him with",
    "at that moment the door",
]


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str):
    """Load model and vocabulary with persistent caching."""
    model, device = load_checkpoint(model_path)
    stoi, itos = load_vocabulary(VOCAB_PATH, ITOS_PATH)
    return model, stoi, itos, device


# Header
st.markdown('<div class="main-title">📖 Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A Neural Language Model that learns language patterns from Leo Tolstoy\'s <i>War and Peace</i> to predict the next word in a sentence.</div>',
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.header("Model & Settings")

# Model Selection
selected_model_name = st.sidebar.selectbox("Choose Trained Model:", list(MODELS.keys()))
model_info = MODELS[selected_model_name]
st.sidebar.caption(model_info["desc"])

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Settings")

words_to_predict = st.sidebar.slider(
    "Number of words to generate:",
    min_value=1,
    max_value=20,
    value=5,
    help="How many subsequent words the model should predict sequentially.",
)

temperature = st.sidebar.slider(
    "Creativity (Temperature):",
    min_value=0.2,
    max_value=1.5,
    value=0.8,
    step=0.1,
    help="Lower values make the model pick the most probable words. Higher values make predictions more diverse.",
)

seed_val = st.sidebar.number_input("Random Seed:", min_value=1, max_value=9999, value=42)

# Load Selected Model
with st.spinner("Loading neural network model..."):
    model, stoi, itos, device = get_cached_model(model_info["path"])

context_size = getattr(model, "size", getattr(model, "context_size", 5))
vocab_size = len(stoi)
total_params = sum(p.numel() for p in model.parameters())

# Sidebar Stats
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write(f"- **Vocabulary Size**: {vocab_size:,} words")
st.sidebar.write(f"- **Context Size**: {context_size} previous words")
st.sidebar.write(f"- **Embedding Size**: {model_info['embed_dim']} dimensions")
st.sidebar.write(f"- **Hidden Layer**: {model_info['hidden_dim']} neurons ({model_info['activation']})")
st.sidebar.write(f"- **Total Parameters**: {total_params:,}")
st.sidebar.write(f"- **Device**: `{device}`")


# Main Tabs
tab_predict, tab_how_it_works, tab_project_info = st.tabs([
    "Try Prediction",
    "How It Works (Simple Explanation)",
    "Project Details & Viva Q&A",
])

# ----------------- TAB 1: PREDICTION -----------------
with tab_predict:
    st.subheader("1. Enter a sentence or pick an example:")

    # Quick example buttons
    cols = st.columns(len(SAMPLE_PROMPTS))
    selected_sample = None
    for i, sample in enumerate(SAMPLE_PROMPTS):
        if cols[i].button(sample, key=f"btn_{i}"):
            st.session_state["user_prompt"] = sample

    default_prompt = st.session_state.get("user_prompt", SAMPLE_PROMPTS[0])
    prompt_input = st.text_input("Starting Words (Context):", value=default_prompt)

    col_btn, col_stats = st.columns([1, 3])
    predict_clicked = col_btn.button("Predict Next Word", type="primary")

    if predict_clicked or "last_result" in st.session_state:
        if predict_clicked:
            if not prompt_input.strip():
                st.error("Please enter at least one word.")
            else:
                generator = TextGenerator(
                    model=model,
                    stoi=stoi,
                    itos=itos,
                    device=device,
                    context_size=context_size,
                    domain="text",
                )

                with st.spinner("Calculating word probabilities..."):
                    result = generator.generate(
                        prompt=prompt_input,
                        max_new_tokens=words_to_predict,
                        temperature=temperature,
                        seed=seed_val,
                        track_probabilities=True,
                    )
                st.session_state["last_result"] = result
                st.session_state["last_prompt"] = prompt_input

        if "last_result" in st.session_state:
            res = st.session_state["last_result"]

            st.markdown("---")
            st.subheader("2. Prediction Result:")

            # Highlight generated words
            orig_prompt = res["prompt"].strip()
            highlighted_words = " ".join([f'<span class="generated-word">{w}</span>' for w in res["generated_tokens"]])
            full_html = f'<div class="prompt-box">{orig_prompt} {highlighted_words}</div>'
            st.markdown(full_html, unsafe_allow_html=True)
            st.caption("The original prompt is in black, and the model's predicted next words are highlighted in blue.")

            # Top candidates for the immediate next word
            if res["steps"]:
                first_step = res["steps"][0]
                st.markdown("---")
                st.subheader(f"Top 5 Candidates for the Immediate Next Word:")
                st.caption(f"Given the context '{orig_prompt}', here is what the model considered most likely:")

                candidates_data = []
                for word, prob in first_step["candidates"][:5]:
                    candidates_data.append({"Candidate Word": word, "Probability (%)": round(prob * 100, 2)})

                df_cand = pd.DataFrame(candidates_data)

                col_table, col_chart = st.columns([1, 1.5])
                with col_table:
                    st.dataframe(df_cand, use_container_width=True, hide_index=True)

                with col_chart:
                    chart_df = df_cand.set_index("Candidate Word")
                    st.bar_chart(chart_df)

                # Step-by-step summary
                if len(res["steps"]) > 1:
                    with st.expander("See all generated steps"):
                        step_summary = []
                        for s in res["steps"]:
                            top_choice = s["candidates"][0]
                            step_summary.append({
                                "Step": f"Word #{s['step']}",
                                "Chosen Word": s["candidates"][0][0],
                                "Confidence": f"{s['candidates'][0][1] * 100:.1f}%",
                                "Top 3 Alternatives": ", ".join([f"{w} ({p*100:.1f}%)" for w, p in s["candidates"][1:4]])
                            })
                        st.dataframe(pd.DataFrame(step_summary), use_container_width=True, hide_index=True)


# ----------------- TAB 2: HOW IT WORKS -----------------
with tab_how_it_works:
    st.subheader("How Does Next Word Prediction Work?")
    st.markdown("""
    This project uses a **Neural Probabilistic Language Model** implemented in PyTorch.
    Here is the step-by-step process in simple terms:
    
    ### 1. The Context Window (5 Previous Words)
    Instead of reading an entire book at once, the model looks at the **last 5 words** you typed.
    For example:
    - Input words: `the`, `prince`, `was`, `silent`, `and`
    
    ### 2. Word Embeddings (Converting Words to Numbers)
    Computers cannot understand raw words directly. Each word is mapped to a unique number ID,
    and then converted into a **dense vector of numbers** (e.g., 64 numbers per word).
    - Words with similar meanings (like *king* and *prince*) end up having similar numbers!
    
    ### 3. Hidden Neural Network Layer
    The 5 word vectors are combined and passed through a hidden layer of neurons (128 or 1024 neurons).
    This layer learns the grammatical patterns and relationships between words from the training text.
    
    ### 4. Probability Output (Softmax)
    The model looks at all **19,764 words** in its vocabulary and gives each word a percentage probability
    of being the next word. The word with the highest probability (or sampled based on temperature) is chosen!
    
    ---
    
    ### Why compare ReLU, Tanh, and Sigmoid?
    In this course project, we trained three identical model setups with different **activation functions**:
    1. **ReLU (Model 1)**: Usually trains fastest and allows building larger networks without vanishing gradients.
    2. **Tanh (Model 2)**: Outputs values between -1 and 1, providing stable zero-centered representations.
    3. **Sigmoid (Model 3)**: A classic activation function that compresses values between 0 and 1.
    """)


# ----------------- TAB 3: PROJECT INFO & VIVA Q&A -----------------
with tab_project_info:
    st.subheader("Undergraduate Project Details")
    st.markdown("""
    - **Project Name**: Next Word Predictor using Neural Language Model
    - **Framework**: PyTorch & Python
    - **Interface**: Streamlit Web Dashboard
    - **Dataset**: Leo Tolstoy's novel *War and Peace* (~19,764 distinct vocabulary words)
    - **Context Size**: 5 words
    
    ---
    
    ### Common Project Viva Questions & Simple Answers:
    
    **Q1: What is the main objective of this project?**
    - **Answer:** To build a neural network that predicts the most likely next word in a sentence based on the preceding 5 words, similar to keyboard autocomplete.
    
    **Q2: What is word embedding and why is it used?**
    - **Answer:** Word embedding represents words as continuous vectors of real numbers rather than large, sparse one-hot vectors. It allows the model to capture semantic similarities (e.g., *happy* and *glad* are placed close together in vector space).
    
    **Q3: What does the temperature parameter do?**
    - **Answer:** Temperature controls how 'creative' or 'safe' the predictions are. Low temperature (e.g. 0.3) makes the model greedily pick only the most probable words. High temperature (e.g. 1.2) flattens probabilities, producing more varied and creative words.
    
    **Q4: What is the context window?**
    - **Answer:** The context window is the number of preceding words the model looks at to make a prediction. In our project, the context size is 5 words.
    
    **Q5: What loss function was used to train the model?**
    - **Answer:** Cross-Entropy Loss (Negative Log-Likelihood), which measures how far the model's predicted probability distribution was from the actual next word in the book.
    """)

import streamlit as st
import torch
import torch.nn as nn
import re
import pickle

# ------------------------------------------------------
# Define the model class (MUST be above torch.load)
# ------------------------------------------------------
class Next_Word_Predictor(nn.Module):
    def __init__(self, size, vocab_size, embed_dim, hidden_dim, activation_fn, seed_value):
        super().__init__()
        self.size = size
        self.hyperpams = {
            'size': self.size,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'activation_fn': activation_fn,
            'seed_value': seed_value
        }
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(size * embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        # Choose activation dynamically
        activations = {
            'relu': torch.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        self.activation_fn = activations.get(activation_fn.lower(), torch.relu)

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


# ------------------------------------------------------
# Setup
# ------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model metadata
model_info = {
    "Model 1": {
        "path": "model1_task1.pt",
        "hidden_dim": 1024,
        "activation_fn": "relu",
        "seed_value": 42,
        "embed_dim": 64
    },
    "Model 2": {
        "path": "model2_task1.pt",
        "hidden_dim": 128,
        "activation_fn": "tanh",
        "seed_value": 123,
        "embed_dim": 32
    },
    "Model 3": {
        "path": "model3_task1.pt",
        "hidden_dim": 128,
        "activation_fn": "sigmoid",
        "seed_value": 999,
        "embed_dim": 64
    }
}


@st.cache_resource
def load_model_and_dicts(model_path):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # Load vocab dictionaries from pickle files
    with open("stoi1.pkl", "rb") as f:
        stoi = pickle.load(f)
    with open("itos1.pkl", "rb") as f:
        itos = pickle.load(f)

    return model, stoi, itos


# ------------------------------------------------------
# Function to generate text
# ------------------------------------------------------
def generateNextWord(model, itos, stoi, content, seed_value, k, temperature=1):
    torch.manual_seed(seed_value)
    size = model.size

    predata = content.lower()
    predata = re.sub(r'[^a-zA-Z0-9 \.]', '', predata)
    predata = re.sub(r'\.', ' . ', predata)
    wordsNew = predata.split()
    predata = []

    # Handle unknown words gracefully
    for i in range(len(wordsNew)):
        if wordsNew[i] in stoi:
            predata.append(wordsNew[i])
        else:
            st.warning(f"⚠️ Word '{wordsNew[i]}' not in vocabulary. Skipping it.")

    # Convert to indices
    predata = [stoi.get(w, 0) for w in predata]

    # Pad or truncate to context size
    if len(predata) <= size:
        predata = [0] * (size - len(predata)) + predata
    elif len(predata) > size:
        predata = predata[-size:]

    # Generate words
    for _ in range(k):
        x = torch.tensor(predata).view(1, -1).to(device)
        y_pred = model(x)
        logits = y_pred / temperature
        word1 = torch.distributions.categorical.Categorical(logits=logits).sample().item()
        word = itos[word1]
        content += " " + word
        predata = predata[1:] + [word1]

    return content


# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Next Word Generator (PyTorch)")
st.write("Enter a sentence beginning, choose a model variant, and predict the next few words.")

# Sidebar model selector
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox("Choose Model Variant:", list(model_info.keys()))

# Display selected model info
info = model_info[model_choice]
st.sidebar.markdown(f"""
**🧩 Model Characteristics**
- File: `{info['path']}`
- Hidden Dim: `{info['hidden_dim']}`
- Embedding Dim: `{info['embed_dim']}`
- Activation: `{info['activation_fn']}`
- Seed: `{info['seed_value']}`
""")

# Load selected model
model, stoi, itos = load_model_and_dicts(info['path'])

# Main input controls
content = st.text_input("✍️ Enter the beginning of a sentence:")
k = st.number_input("🔢 Number of words to generate:", min_value=1, max_value=50, value=5)
temperature = st.slider("🔥 Sampling Temperature:", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if st.button("🚀 Generate"):
    if content.strip() == "":
        st.warning("Please enter some starting text.")
    else:
        with st.spinner("Generating text..."):
            result = generateNextWord(model, itos, stoi, content, info['seed_value'], k, temperature)
        st.success("✅ Generation complete!")
        st.subheader("📝 Generated Sentence:")
        st.write(result)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + PyTorch By Siddhesh")
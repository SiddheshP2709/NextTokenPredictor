import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from torch.serialization import safe_globals

# -------------------------------
# Define your Model Architecture
# -------------------------------
class Next_Word_Predictor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super(Next_Word_Predictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        output, hidden = self.lstm(embeds, hidden)
        logits = self.fc(output[:, -1, :])
        return logits, hidden


# -------------------------------
# Load Pretrained Model Function
# -------------------------------
def load_pretrained_model(model_path):
    try:
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {os.path.abspath(model_path)}")
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with safe_globals([Next_Word_Predictor]):
            model = torch.load(model_path, map_location=device, weights_only=False)

        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        return model

    except Exception as e:
        import traceback
        print("❌ Error loading model:", e)
        traceback.print_exc()
        return None


# -------------------------------
# Generate Next Words Function
# -------------------------------
def generate_next_words(model, itos, stoi, content, seed_value, k, temperature=1.0):
    torch.manual_seed(seed_value)
    device = next(model.parameters()).device

    text = content.lower()
    text = re.sub(r"[^a-zA-Z0-9 \.]", "", text)
    words = text.split()

    for _ in range(k):
        # Prepare input tensor
        input_ids = torch.tensor([[stoi.get(w, 0) for w in words[-5:]]], dtype=torch.long).to(device)

        with torch.no_grad():
            logits, _ = model(input_ids)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            next_word = itos.get(next_id, "<unk>")

        words.append(next_word)

    return " ".join(words)

"""Command-line training script for training next token predictor models."""

import argparse
import sys
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from next_token_predictor.models.mlp import MLPNextTokenPredictor
from next_token_predictor.models.lstm import LSTMNextTokenPredictor
from next_token_predictor.tokenization.text_tokenizer import TextTokenizer
from next_token_predictor.tokenization.code_tokenizer import CodeTokenizer
from next_token_predictor.training.dataset import NGramDataset
from next_token_predictor.training.trainer import Trainer
from next_token_predictor.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Train next token prediction models.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--text-file", type=str, default=None, help="Path to raw corpus text file")
    parser.add_argument("--domain", type=str, choices=["text", "code"], default="text")
    parser.add_argument("--architecture", type=str, choices=["mlp", "lstm"], default="mlp")
    parser.add_argument("--context-size", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="checkpoints/custom")

    args = parser.parse_args()

    # Load configuration if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"[*] Loaded configuration from: {args.config}")
        context_size = cfg.get("dataset", {}).get("context_size", args.context_size)
        embed_dim = cfg.get("model", {}).get("embed_dim", args.embed_dim)
        hidden_dim = cfg.get("model", {}).get("hidden_dim", args.hidden_dim)
        batch_size = cfg.get("training", {}).get("batch_size", args.batch_size)
        epochs = cfg.get("training", {}).get("epochs", args.epochs)
        lr = cfg.get("training", {}).get("learning_rate", args.lr)
    else:
        context_size = args.context_size
        embed_dim = args.embed_dim
        hidden_dim = args.hidden_dim
        batch_size = args.batch_size
        epochs = args.epochs
        lr = args.lr

    device = get_device()
    print(f"[*] Using compute device: {device}")

    # Build or load token sequence
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        if args.domain == "code":
            tokenizer = CodeTokenizer()
            tokens = tokenizer.tokenize_code(raw_text)
            unique_tokens = list(dict.fromkeys(tokens))
            stoi = {t: i for i, t in enumerate(unique_tokens)}
            itos = {i: s for s, i in stoi.items()}
            tokenizer.stoi = stoi
            tokenizer.itos = itos
            token_ids = tokenizer.encode(raw_text)
        else:
            tokenizer = TextTokenizer.build_from_text(raw_text)
            token_ids = tokenizer.encode(raw_text)

        vocab_size = len(tokenizer)
        print(f"[*] Built vocabulary of size: {vocab_size:,}")
    else:
        print("[!] No training corpus file provided. Running demonstration synthetic training step.")
        vocab_size = 1000
        token_ids = torch.randint(0, vocab_size, (5000,)).tolist()

    dataset = NGramDataset(token_ids, context_size=context_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if args.architecture == "lstm":
        model = LSTMNextTokenPredictor(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
    else:
        model = MLPNextTokenPredictor(
            vocab_size=vocab_size,
            context_size=context_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    trainer = Trainer(model=model, optimizer=optimizer, device=device)

    print(f"[*] Starting training for {epochs} epochs on {len(dataset):,} samples...")
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_epoch(dataloader)
        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] "
            f"Loss: {metrics['loss']:.4f} | "
            f"PPL: {metrics['perplexity']:.2f} | "
            f"Acc: {metrics['accuracy'] * 100:.2f}%"
        )

    save_path = os.path.join(args.output_dir, f"{args.architecture}_model.pt")
    trainer.save_checkpoint(save_path)
    print(f"[+] Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

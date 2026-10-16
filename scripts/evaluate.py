"""Model evaluation script: computes loss, perplexity, parameter count, and inference latency."""

import argparse
import sys
import os
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from next_token_predictor.inference.loader import load_checkpoint, load_vocabulary
from next_token_predictor.utils.metrics import compute_perplexity, compute_accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate language model performance metrics.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/war_and_peace/model1_task1.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="data/vocabularies/war_and_peace/stoi1.pkl",
        help="Path to vocabulary mapping",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["text", "cpp", "python"],
        default="text",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=500,
        help="Number of synthetic sliding windows to benchmark",
    )

    args = parser.parse_args()

    print(f"[*] Evaluating Checkpoint: {args.model_path}")
    model, device = load_checkpoint(args.model_path)
    stoi, itos = load_vocabulary(args.vocab_path)

    vocab_size = len(stoi)
    context_size = getattr(model, "size", getattr(model, "context_size", 5))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print("\n" + "=" * 50)
    print("MODEL ARCHITECTURE & CAPACITY")
    print("=" * 50)
    print(f"Model Class:           {type(model).__name__}")
    print(f"Device:                {device}")
    print(f"Context Window Size:   {context_size}")
    print(f"Vocabulary Size:       {vocab_size:,}")
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Model Memory Size:     {model_size_mb:.2f} MB")
    print("=" * 50)

    # Benchmark inference latency
    warmup_x = torch.randint(0, min(100, vocab_size), (1, context_size), device=device)
    for _ in range(10):
        _ = model(warmup_x)

    timed_iterations = 200
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(timed_iterations):
            _ = model(warmup_x)
    total_time = time.perf_counter() - start_time
    latency_ms = (total_time / timed_iterations) * 1000

    print("\n" + "=" * 50)
    print("INFERENCE LATENCY BENCHMARK")
    print("=" * 50)
    print(f"Mean Forward Latency:  {latency_ms:.3f} ms per step")
    print(f"Throughput:            {1000 / latency_ms:.1f} tokens/sec")
    print("=" * 50)


if __name__ == "__main__":
    main()

"""Command-line interface for autoregressive next token prediction."""

import argparse
import sys
import os

# Add src to path for direct invocation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from next_token_predictor.inference.loader import load_checkpoint, load_vocabulary
from next_token_predictor.inference.generator import TextGenerator
from next_token_predictor.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(
        description="Generate autoregressive sequence continuations using trained language models."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/war_and_peace/model1_task1.pt",
        help="Path to trained PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="data/vocabularies/war_and_peace/stoi1.pkl",
        help="Path to vocabulary mapping file (.pkl or .csv)",
    )
    parser.add_argument(
        "--itos-path",
        type=str,
        default=None,
        help="Path to index-to-string mapping file (.pkl, optional)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["text", "python"],
        default="text",
        help="Input domain (affects tokenization rules)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="the prince was silent and",
        help="Seed prompt to continue",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=10,
        help="Number of new tokens to predict",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (lower = more deterministic, higher = more diverse)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K candidates to filter logits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--track-probs",
        action="store_true",
        help="Print top probability candidates at each step",
    )

    args = parser.parse_args()

    print(f"[*] Loading model from: {args.model_path}")
    model, device = load_checkpoint(args.model_path)
    print(f"[*] Model loaded successfully onto {device}.")

    print(f"[*] Loading vocabulary from: {args.vocab_path}")
    stoi, itos = load_vocabulary(args.vocab_path, args.itos_path)
    print(f"[*] Vocabulary size: {len(stoi):,} tokens.")

    generator = TextGenerator(
        model=model,
        stoi=stoi,
        itos=itos,
        device=device,
        domain=args.domain,
    )

    print(f"\n[*] Generating {args.tokens} tokens with prompt: \"{args.prompt}\"")
    result = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        track_probabilities=args.track_probs,
    )

    print("\n" + "=" * 60)
    print("OUTPUT RESULT:")
    print("=" * 60)
    print(result["full_text"])
    print("=" * 60)

    if args.track_probs and result["steps"]:
        print("\nStep-by-Step Probability Analysis:")
        for step_info in result["steps"]:
            cands = ", ".join([f"'{w}': {p*100:.1f}%" for w, p in step_info["candidates"]])
            print(f"  Step {step_info['step']}: {cands}")


if __name__ == "__main__":
    main()

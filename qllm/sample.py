"""Sample text from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from qllm.model import generate, load_model
from qllm.train import DEFAULT_CHECKPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint = DEFAULT_CHECKPOINT
    if not checkpoint.exists():
        checkpoint = checkpoint.with_name("model.pt")
    parser.add_argument("--checkpoint", type=Path, default=checkpoint)
    parser.add_argument("--prompt", default="the ")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, stoi, itos = load_model(args.checkpoint)
    continuation = generate(
        model,
        stoi,
        itos,
        args.prompt,
        args.length,
        temperature=args.temperature,
        greedy=args.greedy,
    )
    print(args.prompt + continuation)


if __name__ == "__main__":
    main()

"""Train the quantum language model on the tiny corpus and write a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from qllm.data import DEFAULT_CORPUS, ROOT, build_vocab, load_text, make_examples
from qllm.model import QuantumLanguageModel, generate, save_checkpoint

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "model.pt"


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=-1)
    return float((predictions == targets).float().mean().item())


def train(args: argparse.Namespace) -> None:
    text = load_text(args.corpus)
    stoi, itos = build_vocab(text)
    features, targets = make_examples(text, stoi, args.context)

    torch.manual_seed(args.seed)
    model = QuantumLanguageModel(
        context_len=args.context,
        n_layers=args.layers,
        vocab_size=len(itos),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    quantum_params, classical_params = model.parameter_count()

    print(
        f"symbols={len(itos)} qubits={model.n_qubits} context={args.context} "
        f"layers={args.layers} quantum_parameters={quantum_params} "
        f"readout_parameters={classical_params} examples={len(targets)}"
    )
    print("alphabet:", "".join(char if char != "\n" else "\\n" for char in itos))

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                acc = accuracy(logits, targets)
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  accuracy {acc:.1%}")

    save_checkpoint(args.checkpoint, model, stoi, itos)
    print(f"saved {args.checkpoint}")

    model.eval()
    prompt = args.prompt
    print()
    print(f"greedy continuation of {prompt!r}:")
    print(prompt + generate(model, stoi, itos, prompt, args.sample_length, greedy=True))
    print()
    print(f"sample at temperature {args.temperature}:")
    print(
        prompt
        + generate(
            model,
            stoi,
            itos,
            prompt,
            args.sample_length,
            temperature=args.temperature,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--prompt", default="the ")
    parser.add_argument("--sample-length", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

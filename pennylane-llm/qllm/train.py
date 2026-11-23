"""Train the quantum language model on a text corpus and write a checkpoint."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from qllm.data import (
    DEFAULT_CORPUS,
    ROOT,
    build_vocab,
    check_context,
    encode,
    eval_windows,
    load_text,
    random_batch,
    split_ids,
)
from qllm.model import QuantumLanguageModel, generate, save_checkpoint

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "model.pt"


@torch.no_grad()
def evaluate(
    model: QuantumLanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Return (loss, accuracy)."""
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets).item()
    accuracy = (torch.argmax(logits, dim=-1) == targets).float().mean().item()
    return float(loss), float(accuracy)


def train(args: argparse.Namespace) -> None:
    check_context(args.context)
    if args.steps < 1:
        raise ValueError("steps must be at least 1.")
    text = load_text(args.corpus)
    stoi, itos = build_vocab(text)
    ids = encode(text, stoi)
    train_ids, val_ids = split_ids(ids, args.context, args.val_fraction)
    val_set = eval_windows(val_ids, args.context, args.eval_windows) if len(val_ids) else None

    torch.manual_seed(args.seed)
    model = QuantumLanguageModel(
        context_len=args.context,
        n_layers=args.layers,
        vocab_size=len(itos),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    embedding_params, quantum_params, readout_params = model.parameter_count()

    print(
        f"corpus={args.corpus.name} chars={len(ids):,} train={len(train_ids):,} "
        f"val={len(val_ids):,} symbols={len(itos)}"
    )
    print(
        f"qubits={model.n_qubits} layers={args.layers} "
        f"embedding_parameters={embedding_params} quantum_parameters={quantum_params} "
        f"readout_parameters={readout_params}"
    )
    print(f"uniform-guess loss {math.log(len(itos)):.3f}")

    started = time.time()
    running_loss = 0.0
    running_steps = 0
    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = random_batch(train_ids, args.context, args.batch_size)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        running_steps += 1

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            line = f"step {step:5d}  train {running_loss / running_steps:.4f}"
            if val_set is not None:
                val_loss, val_acc = evaluate(model, *val_set)
                line += f"  val {val_loss:.4f}  val_acc {val_acc:.1%}"
            print(f"{line}  ({time.time() - started:.0f}s)", flush=True)
            running_loss = 0.0
            running_steps = 0

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
    parser.add_argument("--context", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-windows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--sample-length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

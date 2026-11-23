"""Measure a checkpoint's next-character loss on fixed held-out windows."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from qllm.data import DEFAULT_CORPUS, encode, eval_windows, load_text, split_ids
from qllm.helios import HeliosRunner
from qllm.model import load_model
from qllm.train import DEFAULT_CHECKPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["ideal", "local-noise", "selene", "nexus"], default="ideal")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--project", help="Nexus project for cloud runs.")
    parser.add_argument("--system", choices=["Helios-1E", "Helios-1", "Helios-2"], default="Helios-1E")
    parser.add_argument("--max-cost", type=float, help="Maximum total estimated execution cost in HQC.")
    args = parser.parse_args()
    if args.windows < 1 or args.shots < 1 or args.seed < 0:
        parser.error("windows and shots must be positive and seed must be non-negative")
    if args.backend == "nexus" and not args.project:
        parser.error("--project is required for Nexus evaluation")
    if args.max_cost is not None and (not math.isfinite(args.max_cost) or args.max_cost <= 0):
        parser.error("--max-cost must be finite and positive")
    return args


@torch.no_grad()
def main() -> None:
    args = parse_args()
    runner = None
    if args.backend in {"selene", "nexus"}:
        runner = HeliosRunner(
            args.shots,
            project=args.project,
            system=args.system if args.backend == "nexus" else None,
            max_cost=args.max_cost,
            seed=args.seed,
        )
    model, stoi, _ = load_model(
        args.checkpoint, noise=float(args.backend == "local-noise"), runner=runner
    )
    try:
        ids = encode(load_text(args.corpus), stoi)
    except KeyError as exc:
        raise SystemExit(f"Corpus character {exc} is absent from the checkpoint vocabulary.") from exc
    _, val_ids = split_ids(ids, model.context_len, args.val_fraction)
    if len(val_ids) <= model.context_len:
        raise SystemExit("The validation split is too short for this checkpoint.")
    inputs, targets = eval_windows(val_ids, model.context_len, args.windows, seed=args.seed)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets).item()
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    backend = args.system if args.backend == "nexus" else args.backend
    shots = f" shots_per_window={args.shots}" if runner is not None else ""
    print(f"backend={backend} qubits={model.n_qubits} windows={len(targets)}{shots}")
    print(f"loss={loss:.4f} nats/char  accuracy={accuracy:.1%}")


if __name__ == "__main__":
    main()

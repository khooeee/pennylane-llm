"""Corpus loading, train/validation split, and window sampling."""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "corpus.txt"
# Upper bound is Helios (98 qubits). Simulation memory still doubles per qubit,
# so only small contexts are practical on a local simulator.
MAX_CONTEXT = 98


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text == "":
        raise ValueError(f"Corpus is empty: {path}")
    return text


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    itos = sorted(set(text))
    stoi = {char: index for index, char in enumerate(itos)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[char] for char in text], dtype=torch.long)


def check_context(context_len: int) -> None:
    if not 1 <= context_len <= MAX_CONTEXT:
        raise ValueError(f"context must be from 1 to {MAX_CONTEXT}.")


def split_ids(
    ids: torch.Tensor,
    context_len: int,
    val_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hold out the tail of the corpus for validation.

    The validation split is empty when it would be too short to hold a single window.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val fraction must be at least 0 and less than 1.")
    n_val = int(len(ids) * val_fraction)
    if n_val <= context_len:
        n_val = 0
    train_ids = ids[: len(ids) - n_val]
    val_ids = ids[len(ids) - n_val :]
    if len(train_ids) <= context_len:
        raise ValueError(
            f"Training split has {len(train_ids)} characters; the context window is {context_len}."
        )
    return train_ids, val_ids


def windows(
    ids: torch.Tensor,
    starts: torch.Tensor,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Context windows of shape (batch, context_len) and next-character targets of shape (batch,)."""
    offsets = torch.arange(context_len)
    return ids[starts[:, None] + offsets], ids[starts + context_len]


def random_batch(
    ids: torch.Tensor,
    context_len: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(ids) - context_len, (batch_size,))
    return windows(ids, starts, context_len)


def eval_windows(
    ids: torch.Tensor,
    context_len: int,
    count: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A fixed set of windows, so validation numbers are comparable across steps."""
    available = len(ids) - context_len
    if available <= count:
        starts = torch.arange(available)
    else:
        generator = torch.Generator().manual_seed(seed)
        starts = torch.randperm(available, generator=generator)[:count]
    return windows(ids, starts, context_len)

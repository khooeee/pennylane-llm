"""Corpus loading and angle encoding for the quantum language model."""

from __future__ import annotations

import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "corpus.txt"
MAX_CONTEXT = 8


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text == "":
        raise ValueError(f"Corpus is empty: {path}")
    return text


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    itos = sorted(set(text))
    stoi = {char: index for index, char in enumerate(itos)}
    return stoi, itos


def char_angle(index: int, vocab_size: int) -> float:
    """Spread character ids evenly across one RY rotation."""
    return math.pi * (index + 0.5) / vocab_size


def make_examples(
    text: str,
    stoi: dict[str, int],
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slide a window over the corpus.

    Features have shape (examples, context_len): one angle per context character.
    Targets are the next character id.
    """
    if not 1 <= context_len <= MAX_CONTEXT:
        raise ValueError(f"context must be from 1 to {MAX_CONTEXT}.")
    ids = [stoi[char] for char in text]
    if len(ids) <= context_len:
        raise ValueError(
            f"Corpus has {len(ids)} characters; the context window is {context_len}."
        )

    vocab_size = len(stoi)
    windows = [
        [char_angle(char_id, vocab_size) for char_id in ids[start : start + context_len]]
        for start in range(len(ids) - context_len)
    ]
    targets = [ids[start + context_len] for start in range(len(ids) - context_len)]
    return torch.tensor(windows, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)


def window_features(char_ids: list[int], vocab_size: int) -> torch.Tensor:
    """Encode one context window as a batch of size 1."""
    angles = [char_angle(char_id, vocab_size) for char_id in char_ids]
    return torch.tensor([angles], dtype=torch.float32)

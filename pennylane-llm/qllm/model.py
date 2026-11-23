"""Hybrid quantum next-character model.

Four characters become four rotation angles, one qubit each. A PennyLane circuit
turns that into a probability distribution over basis states, and a small linear
layer reads those probabilities out as next-character logits.
"""

from __future__ import annotations

from pathlib import Path

import pennylane as qml
import torch
from torch import nn

from qllm.data import window_features


class QuantumLanguageModel(nn.Module):
    def __init__(self, context_len: int, n_layers: int, vocab_size: int):
        super().__init__()
        self.context_len = context_len
        self.n_qubits = context_len
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=self.n_qubits)
        self.weights = nn.Parameter(0.1 * torch.randn(*shape))
        self.readout = nn.Linear(2**self.n_qubits, vocab_size)

        wires = list(range(self.n_qubits))
        device = qml.device("default.qubit", wires=self.n_qubits)

        def circuit(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # features: (batch, n_qubits), one angle per context character
            qml.AngleEmbedding(features, wires=wires, rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=wires)
            return qml.probs(wires=wires)

        self.circuit = qml.QNode(circuit, device, interface="torch", diff_method="backprop")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits over the vocabulary. Shape (batch, vocab_size)."""
        probabilities = self.circuit(features, self.weights)
        return self.readout(probabilities.float())

    def parameter_count(self) -> tuple[int, int]:
        quantum = int(self.weights.numel())
        classical = sum(parameter.numel() for parameter in self.readout.parameters())
        return quantum, int(classical)


@torch.no_grad()
def generate(
    model: QuantumLanguageModel,
    stoi: dict[str, int],
    itos: list[str],
    prompt: str,
    length: int,
    temperature: float = 0.5,
    greedy: bool = False,
) -> str:
    """Continue `prompt` one character at a time."""
    missing = sorted({char for char in prompt if char not in stoi})
    if missing:
        shown = ", ".join(repr(char) for char in missing)
        raise ValueError(f"Prompt contains characters that are not in the corpus: {shown}")
    if temperature <= 0 and not greedy:
        raise ValueError("temperature must be positive unless sampling is greedy.")

    pad = stoi.get(" ", 0)
    ids = [stoi[char] for char in prompt]
    if len(ids) < model.context_len:
        ids = [pad] * (model.context_len - len(ids)) + ids

    pieces: list[str] = []
    for _ in range(length):
        window = ids[-model.context_len :]
        features = window_features(window, model.vocab_size)
        logits = model(features)[0]
        if greedy:
            nxt = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(nxt)
        pieces.append(itos[nxt])
    return "".join(pieces)


def save_checkpoint(
    path: Path,
    model: QuantumLanguageModel,
    stoi: dict[str, int],
    itos: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "n_layers": model.n_layers,
            "context_len": model.context_len,
            "vocab_size": model.vocab_size,
        },
        path,
    )


def load_model(path: Path) -> tuple[QuantumLanguageModel, dict[str, int], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = QuantumLanguageModel(
        context_len=payload["context_len"],
        n_layers=payload["n_layers"],
        vocab_size=payload["vocab_size"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["stoi"], payload["itos"]

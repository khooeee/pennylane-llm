"""Hybrid quantum next-character model.

Each context character gets its own qubit. A learned pair of angles per character
rotates that qubit (RY then RZ) to a point on the Bloch sphere. A PennyLane circuit
of strongly entangling layers processes the state, and a linear layer reads one
Pauli-Z expectation per qubit out as next-character logits. The readout grows with
the qubit count instead of exponentially, so context is not capped by 2**qubits.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pennylane as qml
import torch
from torch import nn

from qllm.data import check_context
from qllm.noise import helios_noise_model

CircuitRunner = Callable[[list[qml.tape.QuantumScript]], torch.Tensor]


class QuantumLanguageModel(nn.Module):
    def __init__(
        self,
        context_len: int,
        n_layers: int,
        vocab_size: int,
        noise: float = 0.0,
        *,
        runner: CircuitRunner | None = None,
        readout_kind: str = "expval-z",
    ):
        super().__init__()
        check_context(context_len)
        self.context_len = context_len
        self.n_qubits = context_len
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.noise = noise
        self.runner = runner
        self.readout_kind = readout_kind
        if readout_kind not in {"expval-z", "probs"}:
            raise ValueError(f"Unknown checkpoint readout: {readout_kind}")
        if readout_kind == "probs" and (runner is not None or noise != 0):
            raise ValueError("Train a Pauli-Z checkpoint for Helios or noisy evaluation.")
        if readout_kind == "probs" and context_len > 8:
            raise ValueError("The legacy probability checkpoint supports up to 8 qubits.")
        if runner is not None and noise != 0:
            raise ValueError("The circuit runner sets its own noise model.")

        self.angles = None
        if readout_kind == "expval-z":
            self.angles = nn.Embedding(vocab_size, 2)
            with torch.no_grad():
                self.angles.weight[:, 0].uniform_(0.0, math.pi)
                self.angles.weight[:, 1].uniform_(0.0, 2 * math.pi)

        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=self.n_qubits)
        self.weights = nn.Parameter(0.1 * torch.randn(*shape))
        feature_count = self.n_qubits if self.angles is not None else 2**self.n_qubits
        self.readout = nn.Linear(feature_count, vocab_size)

        wires = list(range(self.n_qubits))

        def circuit(theta: torch.Tensor, phi: torch.Tensor, weights: torch.Tensor):
            # theta, phi: (batch, n_qubits), one angle pair per context character
            qml.AngleEmbedding(theta, wires=wires, rotation="Y")
            if self.angles is not None:
                qml.AngleEmbedding(phi, wires=wires, rotation="Z")
            qml.StronglyEntanglingLayers(weights, wires=wires)
            if self.angles is None:
                return qml.probs(wires=wires)
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in wires)

        self.make_tape = qml.tape.make_qscript(circuit)
        if runner is not None:
            return
        if noise == 0.0:
            device = qml.device("default.qubit", wires=self.n_qubits)
            self.circuit = qml.QNode(circuit, device, interface="torch", diff_method="backprop")
        else:
            # Density matrices make noisy runs much more expensive than noiseless runs.
            device = qml.device("default.mixed", wires=self.n_qubits)
            base = qml.QNode(circuit, device, interface="torch", diff_method="best")
            self.circuit = qml.add_noise(base, helios_noise_model(noise))

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """Map (batch, context_len) character ids to (batch, vocab_size) logits."""
        if self.angles is None:
            theta = (math.pi * (char_ids.double() + 0.5) / self.vocab_size).float()
            return self.readout(self.circuit(theta, theta, self.weights).float())
        angles = self.angles(char_ids)
        if self.runner is not None:
            tapes = [self.make_tape(row[:, 0], row[:, 1], self.weights) for row in angles]
            return self.readout(self.runner(tapes).to(self.readout.weight))
        raw = self.circuit(angles[..., 0], angles[..., 1], self.weights)
        return self.readout(torch.stack(raw, dim=-1).float())

    def parameter_count(self) -> tuple[int, int, int]:
        """Return (embedding, quantum, readout) parameter counts."""
        embedding = 0 if self.angles is None else self.angles.weight.numel()
        quantum = int(self.weights.numel())
        readout = sum(parameter.numel() for parameter in self.readout.parameters())
        return embedding, quantum, int(readout)


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
        window = torch.tensor([ids[-model.context_len :]], dtype=torch.long)
        logits = model(window)[0]
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
            "readout_kind": model.readout_kind,
            "noise": model.noise,
        },
        path,
    )


def load_model(
    path: Path,
    *,
    noise: float | None = None,
    runner: CircuitRunner | None = None,
) -> tuple[QuantumLanguageModel, dict[str, int], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = QuantumLanguageModel(
        context_len=payload["context_len"],
        n_layers=payload["n_layers"],
        vocab_size=payload["vocab_size"],
        noise=payload.get("noise", 0.0) if noise is None else noise,
        runner=runner,
        readout_kind=payload.get("readout_kind", "probs"),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["stoi"], payload["itos"]

"""Helios-like depolarizing noise for the quantum language model.

Defaults mirror the Quantinuum Helios-1E emulator: 2.5e-05 single-qubit fault
probability and 0.0008 two-qubit fault probability.
"""

from __future__ import annotations

import pennylane as qml
from pennylane.noise import NoiseModel, op_eq

HELIOS_P1 = 2.5e-05
HELIOS_P2 = 0.0008


def helios_noise_model(p1: float = HELIOS_P1, p2: float = HELIOS_P2) -> NoiseModel:
    """Return a depolarizing noise model at the given fault probabilities."""
    if p1 < 0 or p2 < 0:
        raise ValueError("fault probabilities must be non-negative.")

    def _after_single(op, **kwargs) -> None:
        qml.DepolarizingChannel(p1, wires=op.wires)

    def _after_cnot(op, **kwargs) -> None:
        for wire in op.wires:
            qml.DepolarizingChannel(p2, wires=wire)

    return NoiseModel(
        {
            op_eq(qml.RY): _after_single,
            op_eq(qml.RZ): _after_single,
            op_eq(qml.Rot): _after_single,
            op_eq(qml.CNOT): _after_cnot,
        }
    )

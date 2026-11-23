"""Local depolarizing approximation of Helios's physical gate infidelities."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pennylane as qml
from pennylane.noise import NoiseModel, op_eq

ONE_QUBIT_INFIDELITY = 2.5e-5
TWO_QUBIT_INFIDELITY = 7.9e-4


def helios_noise_model(scale: float) -> NoiseModel:
    """Convert average gate infidelity to uniform non-identity Pauli error."""
    p1 = 1.5 * ONE_QUBIT_INFIDELITY * scale
    p2 = 1.25 * TWO_QUBIT_INFIDELITY * scale
    if not math.isfinite(scale) or not 0 <= p1 <= 1 or not 0 <= p2 <= 1:
        raise ValueError("noise scale must give finite Pauli error probabilities in [0, 1].")

    paulis = [qml.matrix(gate(0)) for gate in (qml.Identity, qml.X, qml.Y, qml.Z)]
    kraus = [math.sqrt(1 - p2) * np.eye(4)]
    kraus.extend(
        math.sqrt(p2 / 15) * np.kron(left, right)
        for left, right in list(product(paulis, repeat=2))[1:]
    )

    def _after_single(op, **kwargs) -> None:
        qml.DepolarizingChannel(p1, wires=op.wires)

    def _after_cnot(op, **kwargs) -> None:
        qml.QubitChannel(kraus, wires=op.wires)

    return NoiseModel(
        {
            op_eq(qml.RY): _after_single,
            op_eq(qml.RZ): _after_single,
            op_eq(qml.Rot): _after_single,
            op_eq(qml.CNOT): _after_cnot,
        }
    )

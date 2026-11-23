"""Staggered nearest-neighbor circuit with exact local readout simulation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import pennylane as qml
import torch


@dataclass(frozen=True)
class _Layer:
    rotations: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Plan:
    width: int
    layers: tuple[_Layer, ...]
    outputs: tuple[int, ...]

    def apply(self, theta: torch.Tensor, phi: torch.Tensor, weights: torch.Tensor) -> None:
        wires = range(self.width)
        qml.AngleEmbedding(theta, wires=wires, rotation="Y")
        qml.AngleEmbedding(phi, wires=wires, rotation="Z")
        for index, layer in enumerate(self.layers):
            for wire in layer.rotations:
                qml.Rot(*weights[..., index, wire, :].unbind(-1), wires=wire)
            for pair in layer.pairs:
                qml.CNOT(wires=pair)

    def circuit(self, theta: torch.Tensor, phi: torch.Tensor, weights: torch.Tensor):
        self.apply(theta, phi, weights)
        return tuple(qml.expval(qml.Z(wire)) for wire in self.outputs)


class BrickworkCircuit:
    def __init__(self, width: int, depth: int):
        if depth < 1:
            raise ValueError("The brickwork circuit requires at least one layer.")
        wires = tuple(range(width))
        layers = tuple(
            _Layer(wires, tuple((wire, wire + 1) for wire in range(index % 2, width - 1, 2)))
            for index in range(depth)
        )
        self.full = _Plan(width, layers, wires)

    def _light_cone(self, outputs: tuple[int, ...]) -> tuple[_Plan, tuple[int, ...]]:
        active = set(outputs)
        layers = []
        for layer in reversed(self.full.layers):
            pairs = tuple(pair for pair in layer.pairs if active.intersection(pair))
            for pair in pairs:
                active.update(pair)
            layers.append(_Layer(tuple(sorted(active)), pairs))

        inputs = tuple(sorted(active))
        local = {wire: index for index, wire in enumerate(inputs)}
        local_layers = tuple(
            _Layer(
                tuple(local[wire] for wire in layer.rotations),
                tuple((local[left], local[right]) for left, right in layer.pairs),
            )
            for layer in reversed(layers)
        )
        return _Plan(len(inputs), local_layers, tuple(local[wire] for wire in outputs)), inputs

    def simulator(self, make_node: Callable[[Callable, int], qml.QNode]) -> Callable:
        """Batch translated copies of each circuit fragment on small PennyLane devices."""
        last_pairs = self.full.layers[-1].pairs
        paired_wires = {wire for pair in last_pairs for wire in pair}
        single_wires = [(wire,) for wire in self.full.outputs if wire not in paired_wires]
        measurements = [*last_pairs, *single_wires]

        groups = defaultdict(list)
        for outputs in measurements:
            plan, inputs = self._light_cone(outputs)
            groups[plan].append((inputs, outputs))

        nodes = []
        measured_order = []
        for plan, group in groups.items():
            inputs, outputs = zip(*group)
            nodes.append((make_node(plan.circuit, plan.width), torch.tensor(inputs)))
            measured_order.extend(wire for pair in outputs for wire in pair)
        restore_order = torch.tensor(measured_order).argsort()

        def simulate(theta: torch.Tensor, phi: torch.Tensor, weights: torch.Tensor):
            batch_size = theta.shape[0]
            readouts = []
            for node, inputs in nodes:
                # Combine the window and translated cone axes for PennyLane broadcasting.
                theta_batch = theta[:, inputs].flatten(0, 1)
                phi_batch = phi[:, inputs].flatten(0, 1)
                per_cone = weights[:, inputs, :].transpose(0, 1)
                weight_batch = per_cone.expand(batch_size, -1, -1, -1, -1).flatten(0, 1)
                measured = torch.stack(node(theta_batch, phi_batch, weight_batch), dim=-1)
                readouts.append(measured.reshape(batch_size, -1))
            return torch.cat(readouts, dim=-1)[:, restore_order]

        return simulate

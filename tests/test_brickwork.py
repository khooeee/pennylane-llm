import unittest

import pennylane as qml
import torch

from qllm.brickwork import BrickworkCircuit
from qllm.model import QuantumLanguageModel


class BrickworkTests(unittest.TestCase):
    def test_light_cones_match_full_statevector_values_and_gradients(self):
        torch.manual_seed(7)
        width, depth = 9, 4
        model = QuantumLanguageModel(width, depth, 11, ansatz="brickwork")
        inputs = torch.randint(model.vocab_size, (2, width))
        angles = model.angles(inputs)
        full = BrickworkCircuit(width, depth).full
        reference = qml.QNode(
            full.circuit, qml.device("default.qubit", wires=width),
            interface="torch", diff_method="backprop",
        )
        exact = reference(angles[..., 0], angles[..., 1], model.weights)
        expected = model.readout(torch.stack(exact, dim=-1).float())
        actual = model(inputs)
        torch.testing.assert_close(actual, expected)

        parameters = (model.weights, model.angles.weight)
        for observed, target in zip(
            torch.autograd.grad(actual.square().sum(), parameters),
            torch.autograd.grad(expected.square().sum(), parameters),
        ):
            torch.testing.assert_close(observed, target, atol=2e-6, rtol=2e-5)

    def test_every_wire_trains_at_helios_width(self):
        torch.manual_seed(7)
        model = QuantumLanguageModel(98, 4, 16, ansatz="brickwork")
        inputs = torch.randint(model.vocab_size, (2, 98))
        logits = model(inputs)
        self.assertEqual(logits.shape, (2, 16))
        logits.square().sum().backward()
        self.assertTrue((model.weights.grad.abs().sum(dim=(0, 2)) > 0).all())

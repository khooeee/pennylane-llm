import importlib.util
import unittest
from unittest.mock import patch

import torch

from qllm.helios import HeliosRunner
from qllm.model import QuantumLanguageModel


@unittest.skipUnless(importlib.util.find_spec("guppylang"), "Helios extra is required")
class HeliosTests(unittest.TestCase):
    @torch.no_grad()
    def test_selene_preserves_batched_logits_and_wire_order(self):
        torch.manual_seed(7)
        model = QuantumLanguageModel(6, 4, 10)
        runner = HeliosRunner(8192, seed=7)
        sampled = QuantumLanguageModel(6, 4, 10, runner=runner)
        sampled.load_state_dict(model.state_dict())
        windows = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 8, 3, 1, 2, 0]])
        torch.testing.assert_close(sampled(windows), model(windows), atol=0.03, rtol=0)

    @torch.no_grad()
    def test_full_width_program_reports_one_bit_per_qubit(self):
        width = 98
        model = QuantumLanguageModel(width, 4, 10, runner=HeliosRunner(1), ansatz="brickwork")
        angles = torch.linspace(0.0, 1.0, width)
        tape = model.make_tape(angles, angles, model.weights)
        result = (
            HeliosRunner.program(tape)
            .emulator(n_qubits=width)
            .coinflip_sim()
            .with_shots(1)
            .run()
        )
        (bits,) = result.register_bitstrings()["bits"]
        self.assertEqual(len(bits), width)

    @torch.no_grad()
    def test_nexus_estimates_cost_before_submitting(self):
        import qnexus as qnx
        from hugr.qsystem.result import QsysResult

        runner = HeliosRunner(2, project="qllm-test", system="Helios-1E", max_cost=0.5)
        width = 98
        model = QuantumLanguageModel(width, 4, 2, runner=runner, ansatz="brickwork")
        angles = torch.zeros(width)
        tape = model.make_tape(angles, angles, model.weights)
        result = QsysResult([[("bits", [0, 1] * 49)], [("bits", [1] * width)]])
        with (
            patch.object(qnx.projects, "get_or_create"),
            patch.object(qnx.hugr, "upload"),
            patch.object(qnx.hugr, "cost_confidence", return_value=[(1.0, 1.0)]) as estimate,
            patch.object(qnx, "start_execute_job") as submit,
            patch.object(qnx.jobs, "wait_for"),
            patch.object(qnx.jobs, "results") as results,
        ):
            results.return_value = [unittest.mock.Mock(download_result=lambda: result)]
            with self.assertRaisesRegex(RuntimeError, "exceeds"):
                runner([tape])
            submit.assert_not_called()

            estimate.return_value = [(0.25, 1.0)]
            expected = torch.tensor([[0.0, -1.0] * 49])
            torch.testing.assert_close(runner([tape]), expected)
            kwargs = submit.call_args.kwargs
            self.assertEqual(kwargs["backend_config"].system_name, "Helios-1E")
            simulator = kwargs["backend_config"].emulator_config.simulator
            self.assertIsInstance(simulator, qnx.models.MatrixProductStateSimulator)
            self.assertEqual(kwargs["n_qubits"], width)
            self.assertEqual(kwargs["max_cost"], [0.25])

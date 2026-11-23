"""Finite-shot inference for the model's Pauli-Z readout."""

from __future__ import annotations

import math
from uuid import uuid4

import pennylane as qml
import torch


class HeliosRunner:
    def __init__(
        self,
        shots: int,
        *,
        project: str | None = None,
        system: str | None = None,
        max_cost: float | None = None,
        seed: int = 0,
    ):
        self.shots = shots
        self.project = project
        self.system = system
        self.max_cost = max_cost
        self.seed = seed

    @staticmethod
    def program(tape: qml.tape.QuantumScript):
        """Translate the actual PennyLane circuit into a measured Guppy program."""
        try:
            from guppylang import guppy
            from guppylang.std.builtins import array, comptime, output
            from guppylang.std.quantum import measure_array, qubit
            from pytket import Circuit
        except ImportError as exc:
            raise RuntimeError(
                "Install the Helios extra with uv sync --python 3.12 --extra helios."
            ) from exc

        width = len(tape.wires)
        circuit = Circuit(width)
        (decomposed,), _ = qml.transforms.decompose(
            tape, gate_set={"RY", "RZ", "Rot", "CNOT"}
        )
        for operation in decomposed.operations:
            wires = list(operation.wires)
            angles = [float(qml.math.toarray(value)) / math.pi for value in operation.parameters]
            match operation.name:
                case "RY":
                    circuit.Ry(*angles, *wires)
                case "RZ":
                    circuit.Rz(*angles, *wires)
                case "Rot":
                    phi, theta, omega = angles
                    (wire,) = wires
                    circuit.Rz(phi, wire).Ry(theta, wire).Rz(omega, wire)
                case "CNOT":
                    circuit.CX(*wires)
                case _:
                    raise ValueError(f"Unsupported Helios gate: {operation.name}")

        apply_circuit = guppy.load_pytket("apply_circuit", circuit)

        @guppy
        def measured() -> None:
            qubits = array(qubit() for _ in range(comptime(width)))
            apply_circuit(qubits)
            output("bits", array(item.read() for item in measure_array(qubits)))

        return measured

    @staticmethod
    def _expectations(bitstrings: list[str], shots: int, width: int) -> torch.Tensor:
        if len(bitstrings) != shots or any(len(bits) != width for bits in bitstrings):
            raise RuntimeError(f"Expected {shots} shots with {width} bits each.")
        return torch.tensor(
            [[1.0 if bit == "0" else -1.0 for bit in bits] for bits in bitstrings]
        ).mean(dim=0)

    def __call__(self, tapes: list[qml.tape.QuantumScript]) -> torch.Tensor:
        width = len(tapes[0].wires)
        if self.system is None:
            results = [
                self.program(tape)
                .emulator(n_qubits=width)
                .statevector_sim()
                .with_seed(self.seed + index)
                .with_shots(self.shots)
                .run()
                for index, tape in enumerate(tapes)
            ]
        else:
            results = self._run_nexus(tapes, width, self.system)
        return torch.stack(
            [
                self._expectations(
                    result.register_bitstrings(strict_names=True, strict_lengths=True).get("bits", []),
                    self.shots,
                    width,
                )
                for result in results
            ]
        )

    def _run_nexus(self, tapes: list[qml.tape.QuantumScript], width: int, system: str):
        try:
            import qnexus as qnx
        except ImportError as exc:
            raise RuntimeError(
                "Install the Helios extra with uv sync --python 3.12 --extra helios."
            ) from exc

        project = qnx.projects.get_or_create(self.project)
        job_name = f"qllm-{uuid4().hex[:12]}"
        programs = [
            qnx.hugr.upload(self.program(tape).compile(), name=f"{job_name}-{index}", project=project)
            for index, tape in enumerate(tapes)
        ]
        shots = [self.shots] * len(programs)
        emulating = system == "Helios-1E"
        config = qnx.models.HeliosConfig(
            system_name=system,
            emulator_config=qnx.models.HeliosEmulatorConfig() if emulating else None,
        )
        estimates = qnx.hugr.cost_confidence(
            programs, shots, project=project, system_name="Helios-1" if emulating else system
        )
        costs = [cost for cost, _ in estimates]
        if len(costs) != len(programs) or any(not math.isfinite(cost) or cost < 0 for cost in costs):
            raise RuntimeError("Nexus did not return a valid cost for each circuit.")
        if self.max_cost is not None and sum(costs) > self.max_cost:
            raise RuntimeError(f"Estimated cost {sum(costs):g} HQC exceeds {self.max_cost:g} HQC.")
        job = qnx.start_execute_job(
            programs=programs,
            n_shots=shots,
            backend_config=config,
            name=job_name,
            project=project,
            max_cost=costs,
            n_qubits=width if emulating else [],
        )
        qnx.jobs.wait_for(job)
        results = qnx.jobs.results(job)
        if len(results) != len(programs):
            raise RuntimeError("Nexus did not return a result for each circuit.")
        return [result.download_result() for result in results]

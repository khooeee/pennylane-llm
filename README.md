# pennylane-llm

A toy character-level language model with a PennyLane circuit. It encodes each character on one qubit using learned RY and RZ rotations, applies strongly entangling layers, and maps one Pauli-Z expectation per qubit to next-character logits. The readout grows linearly with context length.

## Train and sample

Install with [uv](https://docs.astral.sh/uv/). Local training supports Python 3.11 and newer.

```bash
uv sync
uv run python -m qllm.train
uv run python -m qllm.sample --prompt "the " --length 40
```

The default corpus is the bundled 92-character toy text. Training holds out its last 10% and saves `artifacts/expval-model.pt`. Sampling also supports the bundled original checkpoint, `artifacts/model.pt`. Train a Pauli-Z checkpoint to use the Helios backends.

For a longer corpus such as [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), provide its path and a prompt from its vocabulary.

```bash
uv run python -m qllm.train --corpus path/to/shakespeare.txt --context 6 --steps 5000
```

## Measure noise and run on Helios

Compare cross-entropy in nats per character and next-character accuracy across backends. Use the same checkpoint, corpus, validation fraction, window count and seed for each run. For a meaningful comparison, use a larger held-out set than the toy corpus provides.

```bash
uv run python -m qllm.evaluate --backend ideal
uv run python -m qllm.evaluate --backend local-noise

uv sync --python 3.12 --extra helios
uv run --python 3.12 --extra helios python -m qllm.evaluate --backend selene --shots 1024
```

- `ideal` computes exact Z expectations with PennyLane.
- `local-noise` adds a one-qubit or joint two-qubit depolarizing channel at each corresponding gate. It uses published average gate infidelities of 0.0025% and 0.079%, converted to Pauli-error probabilities. This approximation covers gate noise. Set `--noise 1` on training to use the same approximation during training. [Helios benchmark](https://www.nature.com/articles/s41586-026-10676-4)
- `selene` runs the exported Guppy circuit locally with finite shots. Each shot measures every qubit in the Z basis.
- `nexus` runs the same Guppy circuits on the Helios-1E noisy emulator or on Helios hardware. Helios-1E includes preparation, measurement, leakage, and memory noise. [Quantinuum noise model](https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/noise_model.html)

Sign in to Nexus from the Python environment and use an accessible project. Nexus estimates costs before execution. Pass `--max-cost` to cap the total estimate in HQC. Choose `--system Helios-1` or `--system Helios-2` to run on hardware your organization can access. [Quantinuum’s Nexus workflow](https://docs.quantinuum.com/workflows/run-on-helios/)

```bash
uv run --python 3.12 --extra helios python -m qllm.evaluate \
  --backend nexus --system Helios-1E --project my-project --windows 4 --shots 1024
```

Helios has 98 fully connected physical qubits, so a context of up to 98 characters fits the inference circuit. Four layers at that width use 392 CNOTs before native compilation. Exact statevector and density-matrix simulation grow exponentially in memory. Start with a small checkpoint for local simulation and Helios-1E. The 98-qubit test compiles and exercises the circuit's measurement output using Selene's coin-flip simulator. A hardware accuracy measurement requires a Nexus run.

Run the Guppy parity and width tests locally with the Helios extra installed.

```bash
uv run --python 3.12 --extra helios python -m unittest discover -s tests -v
```

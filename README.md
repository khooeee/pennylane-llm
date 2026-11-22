# pennylane-llm

A toy character-level language model whose forward pass is a PennyLane circuit.

Each character in a short context window is encoded as a Y-rotation on its own qubit. A stack of strongly entangling layers turns that state into a probability distribution over basis states, and a linear layer reads those probabilities out as next-character logits. Training is ordinary cross-entropy with Adam; gradients flow through the circuit with PennyLane's PyTorch interface.

The default corpus is four lines about cats, bats, mats, and hats. The context window is one qubit per character, so it is capped at 8.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
```

## Train

```bash
uv run python -m qllm.train
```

This fits the model on `data/corpus.txt` for 250 epochs and writes `artifacts/model.pt`. At the end it prints a greedy continuation and a sample at temperature 0.5.

Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--context` | `4` | Characters (and qubits) of context. Must be 1–8. |
| `--layers` | `4` | Strongly entangling layers. |
| `--epochs` | `250` | Training epochs over the full corpus. |
| `--lr` | `0.05` | Adam learning rate. |
| `--prompt` | `the ` | Prompt used for the samples printed after training. |
| `--corpus` | `data/corpus.txt` | Training text. The alphabet is every character that appears in this file. |
| `--checkpoint` | `artifacts/model.pt` | Where to save the model. |

## Sample

```bash
uv run python -m qllm.sample --prompt "the cat "
```

`--greedy` picks the most likely next character. Otherwise sampling uses `--temperature` (default `0.5`). The prompt may only contain characters from the training corpus.

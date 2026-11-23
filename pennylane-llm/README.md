# pennylane-llm

A toy character-level language model whose forward pass is a PennyLane circuit, trained on Shakespeare.

Each character in a short context window gets its own qubit. Every character in the alphabet has a learned pair of angles that rotates its qubit (RY, then RZ) to a point on the Bloch sphere. A stack of strongly entangling layers turns that state into a probability distribution over basis states, and a linear layer reads those probabilities out as next-character logits. Training is cross-entropy on random minibatches with Adam and a cosine learning-rate schedule; gradients flow through the simulated circuit with PennyLane's PyTorch interface.

The default corpus is [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) (`data/shakespeare.txt`, about 1.1M characters, 65 symbols). The last 10% is held out for validation. `data/cats.txt` is the original four-line toy corpus.

## What to expect

With the defaults (6 qubits, 4 layers, about 4.4k parameters, most of them in the readout) training takes about two minutes on a laptop CPU and reaches a validation loss of roughly 2.49 nats per character. For reference:

| Model | Loss (nats/char) |
| --- | --- |
| Uniform guess over 65 symbols | 4.17 |
| Character frequencies only (unigram) | 3.31 |
| Fixed angle encoding (the original model) | ~2.85 |
| **This model, defaults** | **~2.49** |
| Bigram count table (measured on its own training text) | 2.45 |

So it learns roughly bigram-level statistics: common letters, word lengths, spacing, and capitalised speaker names. The output looks like this:

```
ROMEO:
Thaten thes mol hos sorr an thet heil ans be thele as iut sonl than; to mear cos hives ans wa a ronr loth bot sor rurt a nate the siv siu uat lu soun sone thar
```

The limit is the hardware model. Context is one qubit per character, and the readout has 2^qubits inputs, so the context is capped at 8 characters. Simulation cost doubles with each extra qubit.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
```

## Train

```bash
uv run python -m qllm.train
```

This trains on `data/shakespeare.txt` for 5000 steps, logging train and validation loss every 500 steps, and writes `artifacts/model.pt`. At the end it prints a greedy continuation and a sample at temperature 0.5.

To train on the toy corpus instead:

```bash
uv run python -m qllm.train --corpus data/cats.txt --context 4 --steps 300 --lr 0.05 --val-fraction 0 --prompt "the "
```

Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--context` | `6` | Characters (and qubits) of context. Must be 1–8. |
| `--layers` | `4` | Strongly entangling layers. |
| `--steps` | `5000` | Optimiser steps. |
| `--batch-size` | `256` | Windows per step. |
| `--lr` | `0.1` | Peak Adam learning rate (cosine-decayed to 0). |
| `--val-fraction` | `0.1` | Tail of the corpus held out for validation. `0` disables it. |
| `--eval-every` | `500` | Steps between validation reports. |
| `--prompt` | `ROMEO:\n` | Prompt used for the samples printed after training. |
| `--corpus` | `data/shakespeare.txt` | Training text. The alphabet is every character that appears in this file. |
| `--checkpoint` | `artifacts/model.pt` | Where to save the model. |

## Sample

```bash
uv run python -m qllm.sample --prompt "JULIET:
"
```

`--greedy` picks the most likely next character (with this small a model it quickly loops on "the the the"). Otherwise sampling uses `--temperature` (default `0.5`). The prompt may only contain characters from the training corpus.

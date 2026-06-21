# Text Generation using RNN, LSTM & GRU

> Next-word prediction trained on **Billie Jean** by Michael Jackson

---

## Problem Statement

Design and implement a Deep Learning model capable of learning the underlying structure, grammar, and contextual dependencies of a given text corpus to generate coherent text sequences using **Vanilla RNN**, **LSTM**, and **GRU**.

---

## Models Used

| Model | Type | Key Feature |
|-------|------|-------------|
| SimpleRNN | Vanilla RNN | Baseline — struggles with long sequences |
| LSTM | Gated RNN | Forget + Input + Output gates for long-term memory |
| GRU | Gated RNN | Reset + Update gates — faster than LSTM |

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## Project Structure



---

## Experiments (Student Tasks)

| # | Experiment | What Changed |
|---|-----------|-------------|
| 1 | Custom corpus | Replaced default text with Billie Jean lyrics |
| 2 | Embedding 32 → 64 | Richer word representations |
| 3 | Epochs 100 → 200 | Longer training |
| 4 | Hidden units 64 → 128 | More model capacity |
| 5 | Generate 10 words | Tested longer text generation |

---

## Key Findings

- **LSTM** and **GRU** significantly outperform Vanilla RNN on longer sequences
- Increasing embedding dimension improved loss slightly
- More epochs helped convergence but risks overfitting on small corpus
- GRU achieves similar accuracy to LSTM with faster training time

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Run all cells top to bottom
3. No extra installations needed — all libraries come pre-installed in Colab

---

## License

This project is for educational purposes only.

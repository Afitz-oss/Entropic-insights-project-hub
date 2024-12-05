# Shakespeare-Style Text Generator

This project involves training a transformer-based language model to generate text in the style of Shakespeare. The model is built from scratch using PyTorch and trained on a custom dataset encoded at the subword level.

## Project Overview

The project implements a custom transformer model inspired by GPT architecture. The model is trained to predict the next token in a sequence, learning contextual relationships to generate coherent text.

### Features
- **Custom Dataset Handling:** Efficient loading of binary-encoded Shakespeare data for training and validation.
- **Transformer Architecture:** Includes multi-head attention, feed-forward layers, and positional embeddings.
- **Training Loop:** Handles training, validation, and checkpointing for tracking progress.
- **CUDA Support:** Leverages GPU acceleration for faster training.

---

## Model Architecture

- **Embedding Layers:** Subword token and positional embeddings.
- **Transformer Blocks:** Stacked layers with:
  - Multi-Head Self-Attention
  - Feed-Forward Neural Networks
  - Layer Normalization and Residual Connections
- **Output Layer:** Linear projection for vocabulary-size logits.

---

## Hyperparameters

- **Batch Size:** `12`
- **Sequence Length:** `65`
- **Learning Rate:** `3e-4`
- **Number of Epochs:** `10`
- **Dropout:** `0.0`
- **Transformer Settings:**
  - **Embedding Dimension:** `64`
  - **Number of Heads:** `4`
  - **Number of Layers:** `4`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Shakespeare-Trained-LLM.git
   cd Shakespeare-Trained-LLM
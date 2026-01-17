# Transformer Encoder – Autoencoding (Masked Language Model)

## Objective

To understand the working of the Transformer Encoder architecture by implementing self-attention, positional encoding, and autoencoding through a Masked Language Modeling (MLM) task.

---

## Problem Statement

Given an input sentence containing masked tokens, the objective is to reconstruct the missing words using a Transformer Encoder.
The same encoder architecture is used to learn contextual word representations without using recurrence or convolution.

Example:

Input:

```
Transformers use [MASK] attention
```

Output:

```
Transformers use self attention
```
nano train_mlm.py
---

## Key Concepts Covered
# Transformer Encoder – Autoencoding (Masked Language Model)

## Objective

To understand the working of the Transformer Encoder architecture by implementing self-attention, positional encoding, and autoencoding through a Masked Language Modeling (MLM) task.

---

## Problem Statement

Given an input sentence containing masked tokens, the objective is to reconstruct the missing words using a Transformer Encoder.
The same encoder architecture is used to learn contextual word representations without using recurrence or convolution.

Example:

Input:

```
Transformers use [MASK] attention
```

Output:

```
Transformers use self attention
```

---

## Key Concepts Covered

* Self-Attention Mechanism
* Scaled Dot-Product Attention
* Sinusoidal Positional Encoding
* Transformer Encoder Architecture
* Autoencoding using Masked Language Modeling
* Attention Weight Visualization

---

## Architecture Overview

The Transformer Encoder consists of:

1. Token Embedding Layer
2. Positional Encoding
3. Multi-Head Self-Attention
4. Feed Forward Network
5. Residual Connections and Layer Normalization

The model processes the entire input sequence in parallel and captures global contextual relationships between tokens.

---

## Project Structure

```
transformer-encoder-autoencoding/
│
├── attention.py                # Scaled dot-product self-attention
├── positional_encoding.py      # Sinusoidal positional encoding
├── encoder.py                  # Transformer encoder stack
├── train_mlm.py                # Masked Language Model training
├── visualize_attention.ipynb   # Attention heatmap visualization
├── results/                    # Output artifacts
└── README.md
```

---

## Implementation Details

### Self-Attention

Scaled dot-product attention computes similarity between query and key vectors, normalizes the scores using softmax, and applies them to value vectors to capture contextual relevance.

### Positional Encoding

Since Transformers lack recurrence, positional information is injected using sinusoidal functions to preserve word order.

### Autoencoding (MLM)

Certain words in the input sentence are replaced with `[MASK]`.
The encoder learns to predict the correct word at masked positions using contextual information from the entire sentence.

---

## Sample Input and Output

| Input Sentence                    | Predicted Output |
| --------------------------------- | ---------------- |
| Transformers use [MASK] attention | self             |
| Mars is called the [MASK] planet  | red              |

---

## Attention Visualization

The attention weights learned by the encoder are visualized as a heatmap.
This demonstrates how each token attends to other tokens in the sequence, validating the global context modeling capability of self-attention.

(Refer to `visualize_attention.ipynb` for the heatmap output.)

---

## Learning Outcomes

* Understood how self-attention replaces recurrence
* Learned how Transformers capture global dependencies
* Implemented encoder-only Transformer architecture
* Gained practical experience with PyTorch and Jupyter
* Visualized attention weights for interpretability

---

## Conclusion

This experiment successfully demonstrates the effectiveness of Transformer Encoders in learning contextual representations using self-attention and autoencoding. The implementation highlights why Transformers outperform traditional sequence models for NLP tasks.

---

## Tools and Technologies

* Python 3.10
* PyTorch
* Jupyter Notebook
* Matplotlib & Seaborn
* Git & GitHub

* Self-Attention Mechanism
* Scaled Dot-Product Attention
* Sinusoidal Positional Encoding
* Transformer Encoder Architecture
* Autoencoding using Masked Language Modeling
* Attention Weight Visualization

---

## Architecture Overview

The Transformer Encoder consists of:

1. Token Embedding Layer
2. Positional Encoding
3. Multi-Head Self-Attention
4. Feed Forward Network
5. Residual Connections and Layer Normalization

The model processes the entire input sequence in parallel and captures global contextual relationships between tokens.

---

## Project Structure

```
transformer-encoder-autoencoding/
│
├── attention.py                # Scaled dot-product self-attention
├── positional_encoding.py      # Sinusoidal positional encoding
├── encoder.py                  # Transformer encoder stack
├── train_mlm.py                # Masked Language Model training
├── visualize_attention.ipynb   # Attention heatmap visualization
├── results/                    # Output artifacts
└── README.md
```

---

## Implementation Details

### Self-Attention

Scaled dot-product attention computes similarity between query and key vectors, normalizes the scores using softmax, and applies them to value vectors to capture contextual relevance.

### Positional Encoding

Since Transformers lack recurrence, positional information is injected using sinusoidal functions to preserve word order.

### Autoencoding (MLM)

Certain words in the input sentence are replaced with `[MASK]`.
The encoder learns to predict the correct word at masked positions using contextual information from the entire sentence.

---

## Sample Input and Output

| Input Sentence                    | Predicted Output |
| --------------------------------- | ---------------- |
| Transformers use [MASK] attention | self             |
| Mars is called the [MASK] planet  | red              |

---

## Attention Visualization

The attention weights learned by the encoder are visualized as a heatmap.
This demonstrates how each token attends to other tokens in the sequence, validating the global context modeling capability of self-attention.

(Refer to `visualize_attention.ipynb` for the heatmap output.)

---

## Learning Outcomes

* Understood how self-attention replaces recurrence
* Learned how Transformers capture global dependencies
* Implemented encoder-only Transformer architecture
* Gained practical experience with PyTorch and Jupyter
* Visualized attention weights for interpretability

---

## Conclusion

This experiment successfully demonstrates the effectiveness of Transformer Encoders in learning contextual representations using self-attention and autoencoding. The implementation highlights why Transformers outperform traditional sequence models for NLP tasks.

---

## Tools and Technologies

* Python 3.10
* PyTorch
* Jupyter Notebook
* Matplotlib & Seaborn
* Git & GitHub


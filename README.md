# Grammatically-Guided Sparse Attention for Efficient and Interpretable Transformers

**Abstract:** The quadratic complexity of self-attention in Transformer models remains a significant bottleneck for processing long sequences and deploying large language models efficiently. For this approach, there has been significant research into Sparse Attention, and Deepseek Sparse Attention has combined various methods of creating segments of tokens to reduce the time complexity. I introduce a novel approach, Grammatically-Guided Sparse Attention, which constrains attention computations based on the grammatical roles of tokens. By leveraging Parts-of-Speech (POS) tags, attention masks are dynamically generated that enforce linguistically coherent connections between tokens, reducing the computational graph without sacrificing essential linguistic dependencies. I propose and evaluate two masking strategies: a hard mask that strictly allows only predefined grammatical interactions, and a soft mask that biases attention towards these interactions. Our experiments, conducted on the SST-2 sentiment classification task using a DistilBERT-like architecture, demonstrate that Grammatically-Guided Sparse Attention maintains comparable accuracy to full attention while significantly reducing the theoretical computational overhead. Preliminary results show accuracy values of 0.8200 for hard masking and 0.8165 for soft masking, closely matching the 0.8200 of full attention, providing a path towards more efficient, interpretable, and linguistically-informed Transformer architectures.

## Repository Setup

This project explores the effect of grammatically guided sparse attention mechanisms in BERT models. It implements custom attention layers that can enforce hard or soft grammatical constraints based on Part-of-Speech (POS) tags.

## Overview

The core idea is to modify the self-attention mechanism in BERT to prioritize or restrict attention between tokens based on linguistic rules (e.g., Adjectives attending to Nouns).

### Key Components

*   **`grammatically_guided_attention.py`**: Contains the custom `GrammaticallyGuidedBertSelfAttention` class which implements:
    *   **Hard Masking**: Binary masks that allow or block attention based on grammatical rules.
    *   **Soft Masking**: Additive bias to attention scores based on grammatical rules.
*   **`run_experiment.py`**: The main script to run training and evaluation experiments. It patches a pre-trained BERT model with the custom attention layer and fine-tunes it on the SST-2 dataset.

## Setup

1.  Install `uv`:
    ```bash
    pip install uv
    ```

2.  Create and activate a virtual environment:
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

4.  Download the SpaCy English model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

Run the experiment script:

```bash
python run_experiment.py
```

This will compare three strategies:
1.  **None**: Standard BERT attention (Baseline).
2.  **Hard**: Hard grammatical masking.
3.  **Soft**: Soft grammatical bias.

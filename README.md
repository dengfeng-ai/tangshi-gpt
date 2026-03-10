# tangshi-gpt

A character-level GPT model that generates Chinese poetry, trained on ~37,000 Tang dynasty (唐朝) poems.

## Overview

This project implements a decoder-only transformer from scratch using PyTorch. The model learns to generate classical Chinese poems character by character, and can produce new poems given a title as a prompt.

### Architecture

- **Tokenizer**: Character-level tokenizer with special tokens (`<sos>`, `<eos>`, `<sep>`, `<pad>`, `<unk>`)
- **Model**: GPT with multi-head self-attention, feed-forward layers, and residual connections
- **Default config**: 6 layers, 8 heads, 256 embedding dimensions, context length of 256

> **Implementation note:** In the standard transformer, the projection matrices W_q, W_k, W_v each have shape `(d_model, d_model)`. In this implementation, each `SelfAttentionHead` uses separate W_q, W_k, W_v matrices of shape `(d_model, head_size)` where `head_size = d_model // n_head`. The `MultiHeadAttention` module then concatenates the outputs of all heads back to `d_model` dimensions. This is mathematically equivalent to a single `(d_model, d_model)` projection followed by a split, but makes the per-head computation more explicit and easier to understand. In production implementations, a single `(d_model, d_model)` matrix is preferred as it allows the projection for all heads to be computed in one batched operation, which is more efficient on GPUs.

### Data

Tang dynasty poems (唐诗) sourced from the [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) dataset. Each poem is encoded as:

```
<sos>[Title]<sep>[Content]<eos>
```

## Project Structure

```
├── data/                    # Pre-split train/val/test poem JSON files
├── checkpoints/             # Saved model checkpoints
├── src/
│   ├── model.py             # Poem dataclass
│   ├── data_preparation.py  # Data loading from train/val/test splits
│   ├── tokenizer.py         # Character-level tokenizer
│   ├── gpt.py               # Transformer model (SelfAttentionHead, MultiHeadAttention, FeedForward, TransformerLayer, GPT)
│   ├── train.py             # Training loop and checkpoint saving
│   ├── generate.py          # CLI for generating poems from a checkpoint
│   ├── evaluate.py          # Comprehensive evaluation script (perplexity, structure, rhyme, diversity)
│   └── rhyme_utils.py       # Rhyme checking via pypinyin (optional dependency)
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.10+

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training

```bash
python src/train.py
```

The training script will:
1. Load poems from pre-split train/val/test files
2. Build a character-level vocabulary
3. Train the GPT model for 10,000 iterations
4. Save a checkpoint to `checkpoints/`

> **Note:** <br>
> I trained for 10,000 iterations on a single GPU(Tesla T4) which took ~1.5 hours.
>   - If you want to train, you can adjust the hyperparameters in `train.py` (e.g., `max_iters`, `batch_size`, `learning_rate`) to fit your resources and needs.
>   - I shared the trained checkpoint in `checkpoints/checkpoint.pt` for you to generate poems without training.

### Generating Poems

After training, generate poems from a saved checkpoint:

```bash
python src/generate.py checkpoints/<checkpoint>.pt --title "春望"
```

Omit `--title` to generate without a title prompt.

Use `--temperature` to control the randomness of the output (default: `1.0`). Lower values produce more deterministic results, higher values increase diversity:

```bash
python src/generate.py checkpoints/<checkpoint>.pt --title "春望" --temperature 0.8
```

Use `--top-p` for nucleus sampling (default: `1.0`). This restricts sampling to the smallest set of tokens whose cumulative probability exceeds the threshold, filtering out unlikely tokens:

```bash
python src/generate.py checkpoints/<checkpoint>.pt --title "春望" --top-p 0.9
```

Both options can be combined:

```bash
python src/generate.py checkpoints/<checkpoint>.pt --title "春望" --temperature 0.8 --top-p 0.9
```

### Evaluation

Evaluate a checkpoint across five dimensions: test-set perplexity, structural validity, rhyme consistency, generation diversity, and qualitative spot-checks.

```bash
python src/evaluate.py checkpoints/<checkpoint>.pt
```

For a quick perplexity-only run (no generation):

```bash
python src/evaluate.py checkpoints/<checkpoint>.pt --perplexity-only
```

Customize the number of generated samples and sampling parameters:

```bash
python src/evaluate.py checkpoints/<checkpoint>.pt --num-samples 500 --temperature 0.8 --top-p 0.9
```

#### Results

Evaluation of the shared checkpoint (`checkpoints/checkpoint.pt`) on 200 generated poems:

| Metric | Score |
|---|---|
| Test set perplexity | 48.64 |
| Structural validity | 96.5% |
| Rhyme consistency | 53.4% |
| Distinct-1 / 2 / 3 | 0.181 / 0.801 / 0.956 |
| Vocab coverage | 23.7% |
| Self-repetition | 0.6% |

The model reliably produces well-formed Tang poetry structures. Rhyme consistency — the hardest aspect for a character-level model to learn implicitly — is an area for further improvement.

## Online Demo

A live demo of the poem generation can be found at [tangshi-gpt](http://tangshi-gpt-models.s3-website-ap-southeast-1.amazonaws.com). 

<img src="images/送别.png" alt="Demo Screenshot" width="500">

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., the original transformer paper
- [nanochat](https://github.com/karpathy/nanochat) — Andrej Karpathy's nanochat
- [Transformer Model Tutorial in PyTorch: From Theory to Code](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch) — DataCamp

## License

MIT

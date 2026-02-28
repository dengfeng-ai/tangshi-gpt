import torch
import torch.nn as nn
from torch.nn import functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class SelfAttentionHead(nn.Module):
    """A self-attention head"""

    def __init__(self, context_size, d_model, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B for batch size, T for sequence length, d_model for embedding dimension
        B, T, d_model = x.shape

        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)

        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        scores = F.softmax(scores, dim=-1)  # (B, T, T)

        scores = self.dropout(scores)

        v = self.value(x)  # (B, T, head_size)
        out = scores @ v  # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """A layer of multi-head attention consists of several heads of self-attention"""

    def __init__(self, context_size, d_model, n_head, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(context_size, d_model, head_size, dropout)
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer with a non-linearity followed by another linear layer"""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    """Transformer layer with one multi-head self-attention sublayer and one feedforward sublayer"""

    def __init__(self, context_size, d_model, n_head, dropout):
        super().__init__()
        head_size = d_model // n_head
        self.attn = MultiHeadAttention(
            context_size, d_model, n_head, head_size, dropout
        )
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT model with token and position embeddings, multiple transformer layers, and a final linear layer to produce logits for each token in the vocabulary"""

    def __init__(self, vocab_size, context_size, d_model, n_head, n_layer, dropout):
        super().__init__()
        self.context_size = context_size

        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(context_size, d_model)
        self.layers = nn.Sequential(
            *[
                TransformerLayer(context_size, d_model, n_head, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input, targets=None):
        # B for batch size, T for sequence length, d_model for embedding dimension
        # input and targets are both (B, T) tensors of integers

        B, T = input.shape

        token_embedded = self.token_embedding_table(input)  # (B, T, d_model)
        position_encoded = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, d_model)

        x = token_embedded + position_encoded  # (B, T, d_model)
        x = self.layers(x)  # (B, T, d_model)
        x = self.ln(x)  # (B, T, d_model)
        logits = self.proj(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            loss = F.cross_entropy(logits.view(B * T, vocab_size), targets.view(B * T))

        return logits, loss

    def generate(
        self, input, max_new_tokens, temperature=1.0, top_p=1.0, eos_token=None
    ):
        # input is (B, T) tensors of integers
        # Copy the input to avoid modifying the original tensor
        token_ids = input.clone()

        for _ in range(max_new_tokens):
            # Crop token_ids to the last context_size tokens
            cropped_token_ids = token_ids[:, -self.context_size :]

            # Get the predictions
            logits, _ = self(cropped_token_ids)  # (B, T, vocab_size)

            # Get the logits for the last time step for all batches
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                # Shift right so the first token above the threshold is kept
                sorted_mask = (
                    cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                )
                sorted_logits[sorted_mask] = float("-inf")

                # Scatter back to original ordering
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # Sample from the distribution
            next_token_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled token id to the running sequence
            token_ids = torch.cat((token_ids, next_token_id), dim=1)  # (B, T+1)

            # Stop generation if all sequences in the batch have produced <eos>
            if eos_token is not None and (next_token_id == eos_token).all():
                break

        return token_ids

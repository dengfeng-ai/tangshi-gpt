from datetime import datetime
import json
import os
import torch

from gpt import device, GPT
from tokenizer import CharTokenizer
from data_preparation import prepare_data

# ============ Hyperparameters =============
# data
batch_size = 64
context_size = 256

# model
d_model = 256
n_head = 8
n_layer = 6
dropout = 0.2

# training
max_iters = 10000
learning_rate = 3e-4
eval_interval = 500


# ============ Data loading =============
def sample_batch(data):
    start_indices = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i : i + context_size] for i in start_indices])
    y = torch.stack([data[i + 1 : i + context_size + 1] for i in start_indices])
    x, y = x.to(device), y.to(device)
    return x, y


# =========== Evaluate the loss on train and val sets =============
@torch.no_grad()
def estimate_loss(train_data, val_data):
    """Evaluate loss deterministically by sliding over the entire dataset."""
    model.eval()

    results = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total_loss = 0.0
        num_batches = (len(data) - context_size) // (batch_size * context_size)
        num_batches = max(num_batches, 1)
        for i in range(num_batches):
            offset = i * batch_size * context_size
            start_indices = [offset + j * context_size for j in range(batch_size)]
            start_indices = [s for s in start_indices if s + context_size < len(data)]
            if not start_indices:
                break
            x = torch.stack([data[s : s + context_size] for s in start_indices]).to(device)
            y = torch.stack([data[s + 1 : s + context_size + 1] for s in start_indices]).to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
        results[name] = total_loss / max(i + 1, 1)

    model.train()
    return results["train"], results["val"]


# ============ Training Loops =============
def train(model: GPT, train_data, val_data, metrics_path: str):
    # Create an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for iter in range(max_iters):
            # Evaluate the loss on train and val sets
            if iter % eval_interval == 0:
                train_loss, val_loss = estimate_loss(train_data, val_data)
                print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                metrics_file.write(json.dumps({"step": iter, "train_loss": train_loss, "val_loss": val_loss}) + "\n")
                metrics_file.flush()

            # Get batch data and calculate the loss
            x, y = sample_batch(train_data)
            _, loss = model(x, y)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, val_loss = estimate_loss(train_data, val_data)
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}\n")
        metrics_file.write(json.dumps({"step": iter, "train_loss": train_loss, "val_loss": val_loss}) + "\n")


# ============ Save checkpoint =============
def save_checkpoint(model: GPT, tokenizer: CharTokenizer):
    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"checkpoints/tangshi_gpt_{timestamp}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char_to_id": tokenizer.char_to_id,
            "id_to_char": tokenizer.id_to_char,
            "vocab_size": tokenizer.vocab_size,
            "context_size": context_size,
            "d_model": d_model,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


# ============ Main function =============
if __name__ == "__main__":
    print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Prepare the data
    train_poems, val_poems, _ = prepare_data()
    print(f"Poems: {len(train_poems)} train, {len(val_poems)} val")

    # Create the tokenizer and build the vocabulary
    train_text = "".join([p.text() for p in train_poems])
    val_text = "".join([p.text() for p in val_poems])
    full_text = train_text + val_text
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(full_text)

    # Encode the poems into token ids
    train_token_ids = tokenizer.encode("".join([p.train_text() for p in train_poems]))
    val_token_ids = tokenizer.encode("".join([p.train_text() for p in val_poems]))

    # Convert token ids to tensors
    train_data = torch.tensor(train_token_ids, dtype=torch.long)
    val_data = torch.tensor(val_token_ids, dtype=torch.long)

    # Create the model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        context_size=context_size,
        d_model=d_model,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    )
    model = model.to(device)

    # Print model size info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,}")

    # Train the model
    os.makedirs("metrics", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"metrics/train_metrics_{timestamp}.jsonl"
    train(model, train_data, val_data, metrics_path)
    print(f"Metrics saved to {metrics_path}")

    # Save checkpoint
    save_checkpoint(model, tokenizer)

    print(f"Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

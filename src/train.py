from datetime import datetime
import os
import torch

torch.manual_seed(42)

from gpt import GPT
from tokenizer import CharTokenizer
from data_preparation import prepare_data

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

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
eval_iters = 200


# ============ Data loading =============
def sample_batch(data):
    start_indices = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i : i + context_size] for i in start_indices])
    y = torch.stack([data[i + 1 : i + context_size + 1] for i in start_indices])
    x, y = x.to(device), y.to(device)
    return x, y


# =========== Evaluate the loss on train and val sets =============
@torch.no_grad()
def estimate_loss(val_data, train_data):
    model.eval()

    # Calculate average loss for train set
    total_train_loss = 0
    for _ in range(eval_iters):
        X, Y = sample_batch(train_data)
        _, loss = model(X, Y)
        total_train_loss += loss.item()

    # Calculate average loss for val set
    total_val_loss = 0
    for _ in range(eval_iters):
        X, Y = sample_batch(val_data)
        _, loss = model(X, Y)
        total_val_loss += loss.item()

    train_loss = total_train_loss / eval_iters
    val_loss = total_val_loss / eval_iters

    model.train()
    return train_loss, val_loss


# ============ Training Loops =============
def train(model: GPT, train_data, val_data):
    # Create an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # Evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            train_loss, val_loss = estimate_loss(val_data, train_data)
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        # Get batch data and calculate the loss
        x, y = sample_batch(train_data)
        _, loss = model(x, y)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss, val_loss = estimate_loss(val_data, train_data)
    print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}\n")


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
    train_poems, val_poems = prepare_data("data")
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
    train(model, train_data, val_data)

    # Save checkpoint
    save_checkpoint(model, tokenizer)

    print(f"Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

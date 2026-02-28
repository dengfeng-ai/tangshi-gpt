import argparse

import torch

from gpt import GPT
from tokenizer import CharTokenizer
from train import device


def load_checkpoint(checkpoint_path: str) -> tuple[GPT, CharTokenizer]:
    """Load a saved checkpoint and return the model and tokenizer."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tokenizer = CharTokenizer.from_pretrained(
        checkpoint["char_to_id"], checkpoint["id_to_char"]
    )

    model = GPT(
        vocab_size=checkpoint["vocab_size"],
        context_size=checkpoint["context_size"],
        d_model=checkpoint["d_model"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def generate_poem(
    model: GPT,
    tokenizer: CharTokenizer,
    device: torch.device,
    title: str = "",
    max_tokens: int = 500,
) -> str:
    """Generate a poem from the model given an optional title."""
    model.eval()

    start_tokens = [tokenizer.char_to_id["<sos>"]]
    if title:
        start_tokens.extend(tokenizer.encode(title))
        start_tokens.append(tokenizer.char_to_id["<sep>"])

    token_ids = torch.tensor([start_tokens], dtype=torch.long, device=device)

    out = model.generate(token_ids, max_new_tokens=max_tokens)[0].tolist()

    text = tokenizer.decode(out)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate a poem from a saved checkpoint"
    )
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint .pt file")
    parser.add_argument(
        "--title", type=str, default="", help="Optional poem title to condition on"
    )
    args = parser.parse_args()

    # Load the model and tokenizer from the checkpoint
    model, tokenizer = load_checkpoint(args.checkpoint)

    # Generate the poem with title provided as context
    poem = generate_poem(model, tokenizer, device, title=args.title)
    print(poem)


if __name__ == "__main__":
    main()

import argparse

import torch

from gpt import device, GPT
from model import Poem
from tokenizer import CharTokenizer


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
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Poem:
    """Generate a poem from the model given an optional title."""
    model.eval()

    start_tokens = [tokenizer.char_to_id["<sos>"]]
    if title:
        start_tokens.extend(tokenizer.encode(title))
        start_tokens.append(tokenizer.char_to_id["<sep>"])

    token_ids = torch.tensor([start_tokens], dtype=torch.long, device=device)

    eos_token = tokenizer.char_to_id["<eos>"]
    out = model.generate(
        token_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token=eos_token,
    )[0].tolist()

    text = tokenizer.decode(out)

    # Split title and content if <sep> is present, otherwise return the whole text as content
    if "<sep>" in text:
        title, content = text.split("<sep>", 1)
    else:
        title = ""
        content = text

    # Remove start and end tokens
    title = title.replace("<sos>", "").replace("<eos>", "").strip()
    content = content.replace("<sos>", "").replace("<eos>", "").strip()

    return Poem(title=title, content=content)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a poem from a saved checkpoint"
    )
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint .pt file")
    parser.add_argument(
        "--title", type=str, default="", help="Optional poem title to condition on"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (lower = more deterministic, higher = more random)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling threshold (0.0-1.0, lower = less random)",
    )
    args = parser.parse_args()

    # Load the model and tokenizer from the checkpoint
    model, tokenizer = load_checkpoint(args.checkpoint)

    # Generate the poem with title provided as context
    poem = generate_poem(
        model,
        tokenizer,
        device,
        title=args.title,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(poem)


if __name__ == "__main__":
    main()

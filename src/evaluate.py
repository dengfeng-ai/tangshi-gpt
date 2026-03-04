"""Comprehensive evaluation script for tangshi-gpt.

Evaluates across five dimensions:
  1. Test set perplexity
  2. Structural validity of generated poems
  3. Rhyme consistency (optional, requires pypinyin)
  4. Generation diversity
  5. Qualitative spot-check at multiple temperatures

Usage:
    python src/evaluate.py checkpoints/checkpoint.pt
    python src/evaluate.py checkpoints/checkpoint.pt --perplexity-only
    python src/evaluate.py checkpoints/checkpoint.pt --num-samples 500 --temperature 0.8
"""

import argparse
import math
import random
from collections import Counter

import torch

from gpt import device, GPT
from model import Poem
from tokenizer import CharTokenizer
from data_preparation import prepare_data
from generate import load_checkpoint, generate_poem

try:
    from rhyme_utils import HAS_PYPINYIN, check_rhyme_consistency
except ImportError:
    HAS_PYPINYIN = False


# ============ Tier 1: Test Set Perplexity ============


@torch.inference_mode()
def compute_perplexity(
    model: GPT,
    test_poems: list[Poem],
    tokenizer: CharTokenizer,
    context_size: int,
    batch_size: int = 64,
    num_batches: int = 200,
) -> tuple[float, float]:
    """Compute perplexity on the test set using random context windows.

    Returns (average_loss, perplexity).
    """
    test_text = "".join(p.train_text() for p in test_poems)
    token_ids = tokenizer.encode(test_text)
    data = torch.tensor(token_ids, dtype=torch.long)

    total_loss = 0.0
    for _ in range(num_batches):
        start_indices = torch.randint(len(data) - context_size, (batch_size,))
        x = torch.stack(
            [data[i : i + context_size] for i in start_indices]
        ).to(device)
        y = torch.stack(
            [data[i + 1 : i + context_size + 1] for i in start_indices]
        ).to(device)
        _, loss = model(x, y)
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# ============ Tier 2: Structural Validity ============


def analyze_structure(poem: Poem) -> dict:
    """Analyze the structural validity of a poem against Tang poetry rules.

    Each content line is a couplet (e.g. "客念紛無極，春淚倍成行。").
    This function splits on ，/。 to recover individual 句 (half-lines),
    then checks punctuation pattern, line count, and character consistency.
    """
    content = poem.content.strip()
    if not content:
        return {"valid": False, "form": None}

    # Flatten content and parse into 句 by splitting on ，and 。
    flat = content.replace("\n", "")
    phrases = []       # individual 句 texts
    punctuation = []   # the delimiter after each 句
    current = []
    for ch in flat:
        if ch in "，。":
            if current:
                phrases.append("".join(current))
                punctuation.append(ch)
                current = []
        else:
            current.append(ch)

    # Trailing text without punctuation means malformed ending
    has_trailing = len(current) > 0
    n_lines = len(phrases)

    result = {
        "line_count": n_lines,
        "valid_punctuation": False,
        "valid_line_count": False,
        "equal_length": False,
        "chars_per_line": None,
        "valid": False,
        "form": None,
    }

    # Punctuation: should alternate ，。，。...
    expected = ["，" if i % 2 == 0 else "。" for i in range(n_lines)]
    result["valid_punctuation"] = (
        not has_trailing
        and punctuation == expected
        and len(punctuation) == n_lines
    )

    # Line count: 绝句 = 4 句, 律诗 = 8 句
    result["valid_line_count"] = n_lines in (4, 8)

    # Equal character count per 句
    char_counts = [len(p) for p in phrases]
    if char_counts and all(c == char_counts[0] for c in char_counts):
        result["equal_length"] = True
        result["chars_per_line"] = char_counts[0]

    # Combined: all three checks pass and line length is 5 or 7
    result["valid"] = all([
        result["valid_punctuation"],
        result["valid_line_count"],
        result["equal_length"],
        result["chars_per_line"] in (5, 7),
    ])

    # Classify into one of four forms
    if result["valid"]:
        forms = {
            (5, 4): "五言绝句",
            (7, 4): "七言绝句",
            (5, 8): "五言律诗",
            (7, 8): "七言律诗",
        }
        result["form"] = forms.get((result["chars_per_line"], n_lines))

    return result


def extract_rhyme_chars(poem: Poem) -> list[str]:
    """Extract rhyme characters (last char before each 。) from a poem."""
    flat = poem.content.strip().replace("\n", "")
    rhyme_chars = []
    for i, ch in enumerate(flat):
        if ch == "。" and i >= 1:
            rhyme_chars.append(flat[i - 1])
    return rhyme_chars


# ============ Tier 4: Diversity Metrics ============


def compute_distinct_n(texts: list[str], n: int) -> float:
    """Compute distinct-n: ratio of unique n-grams to total n-grams."""
    total_ngrams = []
    for text in texts:
        chars = list(text)
        for i in range(len(chars) - n + 1):
            total_ngrams.append(tuple(chars[i : i + n]))
    if not total_ngrams:
        return 0.0
    return len(set(total_ngrams)) / len(total_ngrams)


def compute_self_repetition(poems: list[Poem]) -> float:
    """Compute average intra-poem bigram repetition rate."""
    rates = []
    for poem in poems:
        chars = list(poem.content.replace("\n", ""))
        if len(chars) < 2:
            continue
        bigrams = [tuple(chars[i : i + 2]) for i in range(len(chars) - 1)]
        counts = Counter(bigrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        rates.append(repeated / len(bigrams))
    return sum(rates) / len(rates) if rates else 0.0


def compute_diversity_metrics(poems: list[Poem], tokenizer: CharTokenizer) -> dict:
    """Compute diversity metrics for a set of poems."""
    texts = [p.content for p in poems]

    # Distinct-n ratios
    distinct_1 = compute_distinct_n(texts, 1)
    distinct_2 = compute_distinct_n(texts, 2)
    distinct_3 = compute_distinct_n(texts, 3)

    # Vocabulary coverage
    all_chars = set()
    for text in texts:
        all_chars.update(text)
    regular_vocab = {ch for ch in tokenizer.char_to_id if not ch.startswith("<")}
    vocab_coverage = len(all_chars & regular_vocab) / len(regular_vocab) if regular_vocab else 0.0

    # Self-repetition
    self_rep = compute_self_repetition(poems)

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "distinct_3": distinct_3,
        "vocab_coverage": vocab_coverage,
        "self_repetition": self_rep,
    }


# ============ Generation Helper ============


def generate_poems_batch(
    model: GPT,
    tokenizer: CharTokenizer,
    titles: list[str],
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[Poem]:
    """Generate one poem per title."""
    poems = []
    for title in titles:
        poem = generate_poem(
            model, tokenizer, device,
            title=title,
            temperature=temperature,
            top_p=top_p,
        )
        poems.append(poem)
    return poems


# ============ Main Evaluation ============


def run_evaluation(
    checkpoint_path: str,
    num_samples: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    perplexity_only: bool = False,
):
    print(f"Loading checkpoint: {checkpoint_path}")
    model, tokenizer = load_checkpoint(checkpoint_path)
    context_size = model.context_size

    print(f"Device: {device}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()

    # Load test data
    _, _, test_poems = prepare_data()
    print(f"Test poems: {len(test_poems)}")
    print()

    # ── Tier 1: Perplexity ──────────────────────────────────────
    print("=" * 60)
    print("Tier 1: Test Set Perplexity")
    print("=" * 60)
    avg_loss, perplexity = compute_perplexity(
        model, test_poems, tokenizer, context_size,
    )
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity:   {perplexity:.2f}")
    print()

    if perplexity_only:
        return

    # ── Generate poems for Tiers 2-4 ───────────────────────────
    print(f"Generating {num_samples} poems for evaluation...")
    test_titles = [p.title for p in test_poems]
    sample_titles = random.choices(test_titles, k=num_samples)
    generated_poems = generate_poems_batch(
        model, tokenizer, sample_titles,
        temperature=temperature, top_p=top_p,
    )
    print(f"  Done. Generated {len(generated_poems)} poems.")
    print()

    # ── Tier 2: Structural Validity ─────────────────────────────
    print("=" * 60)
    print("Tier 2: Structural Validity")
    print("=" * 60)

    structures = [analyze_structure(p) for p in generated_poems]
    total = len(structures)

    n_valid_punct = sum(1 for s in structures if s["valid_punctuation"])
    n_valid_lines = sum(1 for s in structures if s["valid_line_count"])
    n_equal_len = sum(1 for s in structures if s["equal_length"])
    n_valid = sum(1 for s in structures if s["valid"])

    print(f"  Valid punctuation (，/。):  {n_valid_punct}/{total} ({n_valid_punct / total:.1%})")
    print(f"  Valid line count (4 or 8): {n_valid_lines}/{total} ({n_valid_lines / total:.1%})")
    print(f"  Equal line length:         {n_equal_len}/{total} ({n_equal_len / total:.1%})")
    print(f"  Fully valid:               {n_valid}/{total} ({n_valid / total:.1%})")

    form_counts = Counter(s["form"] for s in structures if s["form"])
    if form_counts:
        print()
        print("  Form distribution:")
        for form, count in form_counts.most_common():
            print(f"    {form}: {count} ({count / n_valid:.1%} of valid)")
    print()

    # ── Tier 3: Rhyme Consistency ───────────────────────────────
    print("=" * 60)
    print("Tier 3: Rhyme Consistency")
    print("=" * 60)

    if not HAS_PYPINYIN:
        print("  [skipped] pypinyin not installed.")
        print("  Install with: pip install pypinyin")
    else:
        valid_poems = [p for p, s in zip(generated_poems, structures) if s["valid"]]
        if not valid_poems:
            print("  No structurally valid poems to check.")
        else:
            n_rhyming = 0
            for poem in valid_poems:
                rhyme_chars = extract_rhyme_chars(poem)
                result = check_rhyme_consistency(rhyme_chars)
                if result.get("consistent"):
                    n_rhyming += 1
            print(f"  Rhyme-consistent: {n_rhyming}/{len(valid_poems)} ({n_rhyming / len(valid_poems):.1%})")
            print(f"  (among {len(valid_poems)} structurally valid poems)")
    print()

    # ── Tier 4: Generation Diversity ────────────────────────────
    print("=" * 60)
    print("Tier 4: Generation Diversity")
    print("=" * 60)

    gen_metrics = compute_diversity_metrics(generated_poems, tokenizer)
    print(f"  Generated ({len(generated_poems)} poems):")
    print(f"    Distinct-1:      {gen_metrics['distinct_1']:.4f}")
    print(f"    Distinct-2:      {gen_metrics['distinct_2']:.4f}")
    print(f"    Distinct-3:      {gen_metrics['distinct_3']:.4f}")
    print(f"    Vocab coverage:  {gen_metrics['vocab_coverage']:.1%}")
    print(f"    Self-repetition: {gen_metrics['self_repetition']:.4f}")

    test_sample = random.sample(test_poems, min(num_samples, len(test_poems)))
    ref_metrics = compute_diversity_metrics(test_sample, tokenizer)
    print(f"\n  Test set reference ({len(test_sample)} poems):")
    print(f"    Distinct-1:      {ref_metrics['distinct_1']:.4f}")
    print(f"    Distinct-2:      {ref_metrics['distinct_2']:.4f}")
    print(f"    Distinct-3:      {ref_metrics['distinct_3']:.4f}")
    print(f"    Vocab coverage:  {ref_metrics['vocab_coverage']:.1%}")
    print(f"    Self-repetition: {ref_metrics['self_repetition']:.4f}")
    print()

    # ── Tier 5: Qualitative Spot-Check ──────────────────────────
    print("=" * 60)
    print("Tier 5: Qualitative Spot-Check")
    print("=" * 60)

    curated_titles = ["春望", "秋思", "送別", "月夜", "登高", "江雪", "詠梅", "山行"]
    random_titles = random.sample(test_titles, min(4, len(test_titles)))
    spot_titles = curated_titles + random_titles
    temperatures = [0.7, 0.9, 1.1]

    for title in spot_titles:
        print(f"\n  --- {title} ---")
        for temp in temperatures:
            poem = generate_poem(
                model, tokenizer, device,
                title=title,
                temperature=temp,
                top_p=top_p,
            )
            structure = analyze_structure(poem)

            # Rhyme info for valid poems
            rhyme_info = ""
            if HAS_PYPINYIN and structure["valid"]:
                rhyme_chars = extract_rhyme_chars(poem)
                rhyme_result = check_rhyme_consistency(rhyme_chars)
                if rhyme_result.get("available"):
                    rhyme_tag = "yes" if rhyme_result["consistent"] else "no"
                    chars_str = "/".join(rhyme_result["rhyme_chars"])
                    rhyme_info = f" | rhyme={rhyme_tag} ({chars_str})"

            form_str = structure["form"] or "invalid"
            print(f"  [T={temp}] [{form_str}]{rhyme_info}")
            for line in poem.content.split("\n"):
                if line:
                    print(f"    {line}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a tangshi-gpt checkpoint across multiple dimensions",
    )
    parser.add_argument(
        "checkpoint", type=str, help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--perplexity-only", action="store_true",
        help="Only compute test-set perplexity (skip generation-based metrics)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=200,
        help="Number of poems to generate for evaluation (default: 200)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for generation (default: 1.0)",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0,
        help="Top-p (nucleus) sampling threshold (default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_evaluation(
        args.checkpoint,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        perplexity_only=args.perplexity_only,
    )


if __name__ == "__main__":
    main()

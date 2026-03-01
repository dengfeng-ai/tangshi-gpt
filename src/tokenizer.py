SPECIAL_TOKENS = {
    "<pad>": 0,
    "<sos>": 1,  # start of sequence
    "<eos>": 2,  # end of sequence
    "<unk>": 3,  # unknown character
    "<sep>": 4,  # separator
}


class CharTokenizer:
    """A simple character-level tokenizer with special tokens"""

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.vocab_size: int = 0

        self._built = False  # Flag to indicate if the vocabulary has been built

    @classmethod
    def from_pretrained(
        cls, char_to_id: dict[str, int], id_to_char: dict[int, str]
    ) -> "CharTokenizer":
        """Reconstruct a tokenizer from saved mappings without needing training data."""
        instance = cls.__new__(cls)
        instance.char_to_id = char_to_id
        instance.id_to_char = id_to_char
        instance.vocab_size = len(char_to_id)
        instance._built = True
        return instance

    def build_vocab(self, text: str) -> None:
        """Build vocabulary from text"""
        # Start with special tokens
        self.char_to_id = dict(SPECIAL_TOKENS)

        # Get all unique characters in the text
        chars = sorted(set(text))

        next_id = len(SPECIAL_TOKENS)
        for ch in chars:
            if ch not in self.char_to_id:
                self.char_to_id[ch] = next_id
                next_id += 1

        # Reverse mapping
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        self._built = True

        print(f"Vocabulary built: {self.vocab_size} tokens")
        print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
        print(f"  Characters: {self.vocab_size - len(SPECIAL_TOKENS)}")

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids with special token handling"""
        unk_id = self.char_to_id["<unk>"]
        tokens: list[int] = []

        i = 0
        while i < len(text):
            if text[i] == "<":
                end = text.find(">", i)
                if end != -1:
                    candidate = text[i : end + 1]
                    if candidate in SPECIAL_TOKENS:
                        tokens.append(self.char_to_id[candidate])
                        i = end + 1
                        continue
            tokens.append(self.char_to_id.get(text[i], unk_id))
            i += 1

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token ids into text"""
        chars = [self.id_to_char.get(idx, "<unk>") for idx in ids]
        return "".join(chars)

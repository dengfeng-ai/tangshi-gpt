from dataclasses import dataclass


@dataclass
class Poem:
    """A simple data class to represent a poem with a title and content."""

    title: str
    content: str

    def text(self) -> str:
        """Return the full text of the poem, including title and content."""
        return self.title + self.content

    def train_text(self) -> str:
        """Return the text used for training, which includes special tokens."""
        return f"<sos>{self.title}<sep>{self.content}<eos>"

    def __str__(self) -> str:
        """Return a nicely formatted string representation of the poem."""
        return f"{self.title}\n{self.content}"

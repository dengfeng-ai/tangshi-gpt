import json
import os
import random

from model import Poem


def _load_data(dict_path: str) -> list[Poem]:
    """Load all poems from the given JSON files under the given dictionary path."""
    data = []
    for filename in os.listdir(dict_path):
        if filename.endswith(".json"):
            with open(os.path.join(dict_path, filename), "r", encoding="utf-8") as f:
                poems = json.load(f)
                for poem in poems:
                    title = poem["title"]
                    paragraphs = poem["paragraphs"]
                    content = "\n".join(paragraphs)
                    data.append(Poem(title=title, content=content))
    return data


def _split_data(data: list[Poem], train_ratio=0.9) -> tuple[list[Poem], list[Poem]]:
    """Split the data into training and validation sets."""
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def prepare_data(dict_path: str) -> tuple[list[Poem], list[Poem]]:
    """Load and prepare the data from the given dictionary path."""
    data = _load_data(dict_path)
    random.shuffle(data)
    train_data, val_data = _split_data(data)
    return train_data, val_data


if __name__ == "__main__":
    train_data, val_data = prepare_data("data")
    print(f"Total poems: {len(train_data) + len(val_data)}")
    print(f"Train data: {len(train_data)}")
    print(f"Validation data: {len(val_data)}")

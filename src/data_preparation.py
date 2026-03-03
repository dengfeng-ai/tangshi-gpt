import json

from model import Poem


def _load_data(data_path: str) -> list[Poem]:
    """Load all poems from the given JSON file from the specified path."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        poems = json.load(f)
        for poem in poems:
            title = poem["title"]
            content = poem["content"]
            data.append(Poem(title=title, content=content))

    return data


def prepare_data() -> tuple[list[Poem], list[Poem], list[Poem]]:
    """Load and prepare the data from the train, validation, and test JSON files."""
    train_data = _load_data("data/train_poems.json")
    val_data = _load_data("data/val_poems.json")
    test_data = _load_data("data/test_poems.json")

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_data, val_data, test_data = prepare_data()
    print(f"Total poems: {len(train_data) + len(val_data) + len(test_data)}")
    print(f"Train data: {len(train_data)}")
    print(f"Validation data: {len(val_data)}")
    print(f"Test data: {len(test_data)}")

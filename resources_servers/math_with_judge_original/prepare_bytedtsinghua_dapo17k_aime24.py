
from typing import Any

from datasets import load_dataset


def format_dapo(data: dict[str, str | float | int]) -> dict[str, list[Any] | str]:
    return {
        "responses_create_params": {
            "input": data["prompt"][0]["content"],
        },
        "question": data["prompt"][0]["content"],
        "expected_answer": data["reward_model"]["ground_truth"],
    }


train_ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
val_ds = load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")

# Format the examples, removing original columns
train_formatted = train_ds.map(format_dapo, remove_columns=train_ds.column_names)
val_formatted = val_ds.map(format_dapo, remove_columns=val_ds.column_names)

train_formatted.to_json("data/dapo17k_bytedtsinghua_train.jsonl")
val_formatted.to_json("data/aime24_bytedtsinghua_validation.jsonl")

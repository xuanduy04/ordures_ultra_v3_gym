
import json

from datasets import load_dataset


train_ds = load_dataset("YouJiacheng/DAPO-Math-17k-dedup", split="train")

rows = []
for example in train_ds:
    row = {
        "responses_create_params": {"input": example["prompt"]},
        "question": example["prompt"][0]["content"],
        "expected_answer": example["reward_model"]["ground_truth"],
    }
    rows.append(json.dumps(row) + "\n")


with open("data/dapo17k_train.jsonl", "w") as f:
    f.writelines(rows)

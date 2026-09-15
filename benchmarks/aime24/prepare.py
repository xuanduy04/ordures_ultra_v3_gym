
"""Prepare AIME 2024 benchmark data.

Downloads AIME 2024 problems from HuggingFace and converts them to the
Gym benchmark JSONL format with `question` and `expected_answer` fields.
"""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "aime24_benchmark.jsonl"

# HuggingFace dataset for AIME 2024
HF_REPO_ID = "HuggingFaceH4/aime_2024"


def prepare() -> Path:
    """Download and prepare AIME 2024 data. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading AIME 2024 data from {HF_REPO_ID}...")
    ds = load_dataset(HF_REPO_ID, split="train")

    count = 0
    with open(OUTPUT_FPATH, "w") as f:
        for row in ds:
            out = {
                "question": row["problem"],
                "expected_answer": str(row["answer"]),
            }
            f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()

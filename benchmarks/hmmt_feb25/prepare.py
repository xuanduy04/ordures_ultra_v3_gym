
"""Prepare HMMT Feb 2025 benchmark data.

Downloads HMMT Feb 2025 problems from HuggingFace and converts them to the
Gym benchmark JSONL format with `question` and `expected_answer` fields.
Mirrors NeMo Skills' `nemo_skills/dataset/hmmt_feb25/prepare.py`.
"""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "hmmt_feb25_benchmark.jsonl"

HF_REPO_ID = "MathArena/hmmt_feb_2025"


def prepare() -> Path:
    """Download and prepare HMMT Feb 2025 data. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading HMMT Feb 2025 data from {HF_REPO_ID}...")
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

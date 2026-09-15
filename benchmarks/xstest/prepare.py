
"""Prepare XSTest benchmark data."""

import json
from pathlib import Path

from pandas import read_csv


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "xstest_benchmark.jsonl"


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = read_csv("https://raw.githubusercontent.com/paul-rottger/xstest/refs/heads/main/xstest_prompts.csv")

    with OUTPUT_FPATH.open("w") as f:
        for _, row in df.iterrows():
            sample = {
                "responses_create_params": {"input": [{"role": "user", "content": row["prompt"]}]},
                "verifier_metadata": {
                    "label": row["label"],
                    "type": row["type"],
                },
            }
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(df)} samples to {OUTPUT_FPATH}")

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()

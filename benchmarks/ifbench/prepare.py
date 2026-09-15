

"""Prepare IFBench evaluation data for NeMo Gym.

Downloads IFBench test data from AllenAI's GitHub and converts to Gym JSONL
format compatible with the instruction_following resources server.
"""

import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "ifbench_benchmark.jsonl"
RAW_FPATH = DATA_DIR / "IFBench_test.jsonl"
URL = "https://raw.githubusercontent.com/allenai/IFBench/refs/heads/main/data/IFBench_test.jsonl"


def prepare() -> Path:
    """Download IFBench test data and convert to Gym JSONL format."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading IFBench test data from AllenAI GitHub...")
    urllib.request.urlretrieve(URL, RAW_FPATH)

    rows = []
    with open(RAW_FPATH, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            entry = json.loads(line)
            prompt = entry["prompt"]
            rows.append(
                {
                    "id": idx,
                    "instruction_id_list": entry["instruction_id_list"],
                    "prompt": prompt,
                    "kwargs": entry["kwargs"],
                    "grading_mode": "fraction",
                }
            )

    with open(OUTPUT_FPATH, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()

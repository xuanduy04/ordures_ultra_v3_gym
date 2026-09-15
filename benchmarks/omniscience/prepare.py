

"""Prepare AA-Omniscience evaluation data for NeMo Gym.

Downloads the AA-Omniscience-Public dataset from HuggingFace and converts
to Gym JSONL format compatible with the omniscience resource server.

Output is raw data — no prompts baked in. Prompts are applied at rollout
time via prompt_config (e.g., +prompt_config=benchmarks/omniscience/prompts/generation.yaml).

Source: https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "omniscience_benchmark.jsonl"


def prepare() -> Path:
    """Download AA-Omniscience data and convert to Gym JSONL format."""
    from datasets import load_dataset

    print("Downloading AA-Omniscience-Public from HuggingFace...")
    ds = load_dataset("ArtificialAnalysis/AA-Omniscience-Public", split="train")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in ds:
        row = {
            "id": entry["question_id"],
            "domain": entry["domain"],
            "topic": entry["topic"],
            "question": entry["question"],
            "expected_answer": entry["answer"],
        }
        rows.append(json.dumps(row, ensure_ascii=False) + "\n")

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()

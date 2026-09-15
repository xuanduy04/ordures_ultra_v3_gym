

"""Prepare SimpleQA-Verified evaluation data for NeMo Gym.

Downloads `codelion/SimpleQA-Verified` (the verified subset of OpenAI's
SimpleQA) from HuggingFace and converts to Gym JSONL format compatible with
the simpleqa resource server.

Output is raw data — no prompts baked in. Prompts are applied at rollout
time via `prompt_config=benchmarks/simpleqa/prompts/default.yaml`.

Source: https://huggingface.co/datasets/codelion/SimpleQA-Verified
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "simpleqa_benchmark.jsonl"


def prepare() -> Path:
    """Download SimpleQA-Verified and convert to Gym JSONL format.

    Mirrors Skills' format_entry_verified: id from `original_index`, full
    upstream row preserved as `metadata`, plus the canonical `question` and
    `expected_answer` fields the resource server reads.
    """
    from datasets import load_dataset

    print("Downloading codelion/SimpleQA-Verified from HuggingFace...")
    ds = load_dataset("codelion/SimpleQA-Verified", split="train")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, entry in enumerate(ds):
        # Mirror Skills' prepare.format_entry_verified: id falls back to a
        # stable per-row tag, metadata carries the full upstream row.
        row_id = entry.get("original_index", f"simpleqa_{idx}")
        row = {
            "id": row_id,
            "metadata": dict(entry),
            "question": entry["problem"],
            "expected_answer": entry["answer"],
        }
        rows.append(json.dumps(row, ensure_ascii=False) + "\n")

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()

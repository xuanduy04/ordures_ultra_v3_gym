
"""Prepare HotpotQA closed-book validation set for NeMo Gym.

Same source as NeMo Skills' `hotpotqa_closedbook` benchmark — the
HuggingFace `hotpotqa/hotpot_qa` distractor configuration's validation
split — so the per-row `question` / `expected_answer` strings are
byte-identical between Skills and Gym.

Emits one JSONL row per problem with `question`, `expected_answer`, and
the original metadata fields (`id`, `type`, `level`) that Skills also
preserves. The closed-book formulation drops `context` and
`supporting_facts` since the model answers without external context.
"""

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "hotpotqa_closedbook_benchmark.jsonl"


def prepare() -> Path:
    """Download HotpotQA distractor validation, write closed-book JSONL.

    Returns the output file path.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading hotpotqa/hotpot_qa distractor:validation ...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    tmp_fpath = OUTPUT_FPATH.with_suffix(".jsonl.tmp")
    count = 0
    with tmp_fpath.open("w", encoding="utf-8") as out:
        for entry in tqdm(ds, desc="Writing closed-book JSONL"):
            row = {
                "id": entry["id"],
                "question": entry["question"],
                "expected_answer": entry["answer"],
                "type": entry["type"],
                "level": entry["level"],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp_fpath.replace(OUTPUT_FPATH)

    print(f"Wrote {count} examples to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()

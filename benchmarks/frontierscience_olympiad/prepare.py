
"""Prepare FrontierScience Olympiad data for NeMo Gym.

Downloads ``openai/frontierscience`` (config ``olympiad``, split ``test``)
from HuggingFace — same source as NeMo Skills'
``nemo_skills/dataset/frontierscience-olympiad/prepare.py`` — and writes a
single combined JSONL with chemistry + biology + physics rows. Each row
carries the ``subject`` field through so the resource server can compute
per-subject pass@k breakdowns.

Output: ``benchmarks/frontierscience_olympiad/data/frontierscience_olympiad_benchmark.jsonl``

Source: https://huggingface.co/datasets/openai/frontierscience
"""

import json
import re
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "frontierscience_olympiad_benchmark.jsonl"

HF_REPO = "openai/frontierscience"
HF_SPLIT = "test"
HF_DATA_FILE = "olympiad/test.jsonl"

# Three subjects as enumerated by Skills (chemistry, biology, physics). The
# combined JSONL keeps all three; the per-subject pass@k breakdown is
# computed at metric-aggregation time via compute_subset_metrics.
SUBJECTS = ("chemistry", "biology", "physics")


def _format_entry(entry: dict, problem_index: int, subject: str) -> dict:
    """Convert a HF row to Gym JSONL format. Mirrors Skills' format_entry."""
    # Remove surrounding backticks (handles `, ``, ```, etc.) — Skills does this too.
    answer = re.sub(r"^`+|`+$", "", entry.get("answer", "") or "").strip()

    return {
        "id": f"olympiad-{problem_index}",
        "question": entry.get("problem", ""),
        "expected_answer": answer,
        "subject": subject,
        "task_group_id": entry.get("task_group_id", "") or "",
    }


def prepare() -> Path:
    """Download the dataset and write the combined Gym JSONL."""
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {HF_REPO} ({HF_DATA_FILE}, split={HF_SPLIT}) ...")
    olympiad_data = load_dataset(
        HF_REPO,
        data_files={HF_SPLIT: HF_DATA_FILE},
        split=HF_SPLIT,
    )
    print(f"Loaded {len(olympiad_data)} olympiad problems")

    count = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for idx, entry in enumerate(olympiad_data):
            subject = (entry.get("subject", "") or "").lower()
            if subject not in SUBJECTS:
                continue
            row = _format_entry(entry, idx, subject)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()

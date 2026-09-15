
"""LiveCodeBench v6 (Aug 2024 – May 2025).

Matches Skills' default split (EVAL_SPLIT = "test_v6_2408_2505"). Uses the raw
livecodebench HF dataset with decoded private test cases.
"""

from pathlib import Path

from benchmarks.livecodebench.prepare_utils import prepare_from_hf_raw


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FPATH = DATA_DIR / "livecodebench_v6_validation.jsonl"


def prepare() -> Path:
    return prepare_from_hf_raw(
        OUTPUT_FPATH, release_version="release_v6", date_from="2024-08-01", date_to="2025-06-01"
    )


if __name__ == "__main__":
    prepare()

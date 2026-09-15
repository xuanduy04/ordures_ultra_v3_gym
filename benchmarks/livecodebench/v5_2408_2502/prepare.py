
"""LiveCodeBench v5 (Aug 2024 – Feb 2025, 279 problems).

Matches Skills' test_v5_2408_2502 split. Uses the pre-prepared code_gen validation
dataset from HuggingFace (test cases from the official LCB runner).
"""

from pathlib import Path

from benchmarks.livecodebench.prepare_utils import prepare_from_hf_raw


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FPATH = DATA_DIR / "livecodebench_v5_validation.jsonl"


def prepare() -> Path:
    return prepare_from_hf_raw(
        OUTPUT_FPATH, release_version="release_v5", date_from="2024-08-01", date_to="2025-03-01"
    )


if __name__ == "__main__":
    prepare()


"""Prepare Spider2 Lite benchmark data."""

from argparse import Namespace
from pathlib import Path
from shutil import copy

from resources_servers.spider2_lite.scripts.prepare_dataset import _main, clone_spider2_repo, delete_spider2_repo
from resources_servers.spider2_lite.setup_spider2 import _DEFAULT_DIR, ensure_spider2_lite


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "spider2_lite_benchmark.jsonl"


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download SQL lite databases
    ensure_spider2_lite()

    clone_spider2_repo(parent_dir=_DEFAULT_DIR)

    _main(
        args=Namespace(
            spider2_dir=_DEFAULT_DIR / "Spider2" / "spider2-lite",
            sqlite_dir=None,
            output_dir=str(OUTPUT_FPATH.parent),
        )
    )
    copy(
        OUTPUT_FPATH.parent / "spider2_lite_sqlite_validation.jsonl",
        OUTPUT_FPATH,
    )

    delete_spider2_repo(parent_dir=_DEFAULT_DIR)

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()

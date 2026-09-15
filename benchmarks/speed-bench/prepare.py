
"""SPEED-Bench data preparation for Gym (qualitative config).

Each `type: benchmark` agent in `config.yaml` calls a separate prepare
module via its `prepare_script` field; this one prepares the qualitative
config. See `prepare_throughput_2k.py` for the throughput_2k variant. Both
share helpers in `_prepare_common.py`.

The Skills `_resolve_external_data` function is loaded by file path from
the mounted Skills checkout (`/nemo_run/code/nemo_skills/dataset/speed-bench/prepare.py`)
so we don't have to re-port the 14-source interpolation logic.
"""

import argparse
import sys
from pathlib import Path


# The benchmark dir name has a hyphen, so importlib's dotted-path import
# (`benchmarks.speed-bench.prepare`) is not a valid identifier and the
# parent dir isn't auto-added to sys.path. Prepend it so the sibling
# `_prepare_common` module is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _prepare_common import prepare_one_config  # noqa: E402


def prepare() -> Path:
    """Prepare the qualitative split. Returns the output JSONL path."""
    return prepare_one_config("qualitative")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SPEED-Bench qualitative data for Gym.")
    parser.parse_args()
    prepare()

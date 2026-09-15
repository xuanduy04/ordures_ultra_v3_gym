
"""SPEED-Bench data preparation for Gym (throughput_2k config).

See `prepare.py` for the qualitative variant. Both delegate to
`_prepare_common.prepare_one_config`.
"""

import argparse
import sys
from pathlib import Path


# See prepare.py for why we manually extend sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _prepare_common import prepare_one_config  # noqa: E402


def prepare() -> Path:
    """Prepare the throughput_2k split. Returns the output JSONL path."""
    return prepare_one_config("throughput_2k")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SPEED-Bench throughput_2k data for Gym.")
    parser.parse_args()
    prepare()

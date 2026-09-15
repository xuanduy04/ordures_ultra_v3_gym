
"""Prepare finance_sec_search benchmark with web_search tool included.

Thin wrapper around prepare.prepare() — used by config_web_search.yaml
so ng_prepare_benchmark produces the web_search variant.
"""

from pathlib import Path

from benchmarks.finance_sec_search.prepare import prepare as _prepare


def prepare() -> Path:
    return _prepare(include_web_search=True)


if __name__ == "__main__":
    prepare()

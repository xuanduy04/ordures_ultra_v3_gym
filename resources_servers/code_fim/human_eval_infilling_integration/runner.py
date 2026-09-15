

"""Ray-remote wrapper for HumanEval-Infilling per-task verification.

Wraps `human_eval_infilling.execution.check_correctness` so that each
verify() call in the resource server can dispatch a single-task check to
a Ray worker and await the result without blocking the asyncio loop.

The Ray runtime_env mirrors the pattern used by
`resources_servers/evalplus/evalplus_integration/runner.py`: the worker
uses the server's venv Python and adds the server directory to
PYTHONPATH so the package-less `human_eval_infilling_integration`
module resolves.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import ray


# human_eval_infilling_integration/ has no pyproject.toml so the worker
# needs the server dir on PYTHONPATH for `import
# human_eval_infilling_integration` to resolve.
_SERVER_DIR = str(Path(__file__).parent.parent)


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
        "env_vars": {"PYTHONPATH": _SERVER_DIR},
    },
)
def check_correctness_remote(
    problem: Dict[str, Any],
    completion: str,
    timeout: float,
) -> Dict[str, Any]:
    """Run the FIM check for a single (task, completion) pair.

    Args mirror `human_eval_infilling.execution.check_correctness`:
    the ``problem`` dict must contain at least ``task_id``, ``prompt``
    (the prefix), ``suffix``, ``test`` and ``entry_point``. The check
    program executed by the upstream library is::

        problem["prompt"] + completion + problem["suffix"] + "\n" +
        problem["test"] + "\n" + f"check({problem['entry_point']})"

    Returns a dict with ``passed: bool`` and ``result: str``
    (``"passed" | "timed out" | f"failed: {exc}"``).
    """
    from human_eval_infilling.execution import check_correctness

    result = check_correctness(
        problem=problem,
        completion=completion,
        timeout=timeout,
        completion_id=0,
    )
    return {
        "passed": bool(result.get("passed", False)),
        "result": result.get("result"),
    }

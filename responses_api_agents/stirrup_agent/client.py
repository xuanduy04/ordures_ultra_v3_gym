
"""Minimal client for the Stirrup agent wrapper.

Reads the first task from ``data/example.jsonl`` and POSTs it to the
Stirrup agent's ``/v1/responses`` endpoint. Prints the resulting
deliverable text and reward.

Run the Stirrup + model servers first (see README.md), then::

    python responses_api_agents/stirrup_agent/client.py
"""

import json
from asyncio import run
from pathlib import Path

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


def main() -> None:
    example_path = Path(__file__).parent / "data" / "example.jsonl"
    with example_path.open() as f:
        first_line = next(f)
    record = json.loads(first_line)
    params_dict = record["responses_create_params"]

    server_client = ServerClient.load_from_global_config()
    task = server_client.post(
        server_name="stirrup_agent",
        url_path="/v1/responses",
        json=NeMoGymResponseCreateParamsNonStreaming(**params_dict),
    )
    response = run(task)
    print(json.dumps(run(response.json()), indent=2))


if __name__ == "__main__":
    main()


import json
from asyncio import run

from nemo_gym.server_utils import ServerClient


async def main():
    server_client = ServerClient.load_from_global_config()
    result = await server_client.post(
        server_name="harbor_agent",
        url_path="/run",
        json={
            "responses_create_params": {
                "input": [],
            },
            "instance_id": "scientific::scientific_computing_task_0001",
        },
    )
    data = await result.json()
    print(json.dumps(data, indent=4))


run(main())

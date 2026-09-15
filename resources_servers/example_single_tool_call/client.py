
import json
from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="example_single_tool_call",
    url_path="/get_weather",
    json={
        "city": "San Francisco, CA",
    },
)
result = run(task)
print(json.dumps(run(result.json()), indent=4))

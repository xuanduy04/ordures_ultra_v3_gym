
import json
from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="math_advanced_calculations",
    url_path="/add",
    json={
        "a": 1,
        "b": 3,
    },
)
result = run(task)
print(json.dumps(run(result.json()), indent=4))

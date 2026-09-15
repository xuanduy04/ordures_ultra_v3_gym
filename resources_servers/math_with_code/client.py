
import json
from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="math_with_code",
    url_path="/execute_python",
    json={
        "code": """
print("Hello, world!")
a=np.ones(10)
print(np.sum(a))
"""
    },
)
result = run(task)
print(json.dumps(run(result.json()), indent=4))

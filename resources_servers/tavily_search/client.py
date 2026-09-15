
import json
from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="tavily_search_resources_server",
    url_path="/search",
    json={
        "query": "Who is nobel laureate in 2025 for economics? look up the answer on the internet and return the answer.",
    },
)
result = run(task)
print(json.dumps(run(result.json()), indent=4))

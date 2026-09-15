
import json

from app import VlmEvalKitResourcesServer


# From W&B table
fpath = ""
with open(fpath) as f:
    table = json.load(f)

rows = [json.loads(row[0]) | {"benchmark_name": "MMBench_DEV_EN_V11"} for row in table["data"]]

aggregate_metrics = VlmEvalKitResourcesServer._aggregate_MMBench_DEV_EN_V11(None, [rows])
print(json.dumps(aggregate_metrics, indent=4))

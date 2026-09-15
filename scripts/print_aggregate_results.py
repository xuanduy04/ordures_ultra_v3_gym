

import json
from collections import Counter

from pydantic import BaseModel

from nemo_gym.global_config import get_global_config_dict


class PrintAggregateResultsConfig(BaseModel):
    jsonl_fpath: str


if __name__ == "__main__":
    global_config_dict = get_global_config_dict()
    config = PrintAggregateResultsConfig.model_validate(global_config_dict)

    metrics = Counter()
    num_rows = 0
    with open(config.jsonl_fpath) as f:
        for row in f:
            result = json.loads(row)
            metrics.update({k: v for k, v in result.items() if isinstance(v, (int, float))})
            num_rows += 1

    avg_metrics = {k: v / num_rows for k, v in metrics.items()}

    print(json.dumps(avg_metrics, indent=4))

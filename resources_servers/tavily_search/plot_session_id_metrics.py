
"""
Usage:
```bash
python resources_servers/tavily_search/plot_session_id_metrics.py \
    --fpath resources_servers/tavily_search/session_id_metrics.json
```
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from pandas import DataFrame


parser = ArgumentParser()
parser.add_argument("--fpath", type=str, required=True)
args = parser.parse_args()

with open(args.fpath) as f:
    data = json.load(f)

rows = []
for session_id_metrics in data.values():
    rows.extend(session_id_metrics["async_tavily_calls"])

df = DataFrame.from_records(rows)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.hist(df["time_taken"])
ax.set_xlabel("Time taken per call (s)")
ax.set_ylabel("Count")
ax.set_title(f"Call time distribution (total {len(df)} Tavily API calls)")

fig.tight_layout()
out_path = Path(__file__).parent / "metrics.png"
fig.savefig(out_path)

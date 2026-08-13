# MultiChallenge Environment

Evaluates model responses on the MultiChallenge benchmark (multi-turn
conversations with rubric-based yes/no criteria) using an LLM judge whose
hosting is **not managed by NeMo-Gym**. The YAML config supplies a
`judge_server_url` (`host:port` of an already-running `vllm serve` endpoint) and
a `judge_model` name. The judge is queried via the OpenAI **Chat Completions**
API (`{judge_server_url}/v1/chat/completions`), which is the surface a stock
`vllm serve` exposes.

All rubric / aggregation / verdict / data-schema logic is inherited unchanged
from [`resources_servers/multichallenge_original`](../multichallenge_original),
which is the original Gym-managed-judge variant.

## How it differs from `multichallenge_original`

| Aspect | `multichallenge_original` | `multichallenge` (this) |
|--------|---------------------------|-------------------------|
| Judge hosting | NeMo-Gym manages a `responses_api_models` vLLM server | Externally hosted; YAML supplies `judge_server_url` |
| Judge protocol | `/v1/responses` (Responses API) via `ServerClient` | `/v1/chat/completions` (native vLLM) via direct HTTP |
| Config fields | `judge_model_server` (a `ModelServerRef`) | `judge_server_url` + `judge_model` (both mandated, no defaults) |
| Init validation | (none — Gym spins up the judge) | `judge_server_url` normalized + `judge_model` checked against `/v1/models` |
| Rubric / aggregation / verdict / data schema | — | **Identical** (inherited from `multichallenge_original`) |

Everything else — per-rubric-item judge calls (`asyncio.gather` when
`parallel_evaluation=True`), `[[YES]]`/`[[NO]]` verdict extraction,
`aggregation_mode` (mean/min/max/all/any/weighted), request/response models,
context/rubric resolution — is inherited unchanged.

## Data compatibility

Any dataset compatible with `multichallenge_simple_agent` (the original variant
under `multichallenge_original/`) works here **unchanged** — the
`agent_ref.name` is the same (`multichallenge_simple_agent`). See
`data/README.md` for details.

## Configuration

```yaml
multichallenge:
  resources_servers:
    multichallenge:
      entrypoint: app.py

      # MANDATED (no defaults):
      judge_server_url: "0.0.0.0:8000"   # host:port of the external vLLM judge
      judge_model: "Qwen/Qwen3-30B-A3B-Instruct-2507"

      judge_responses_create_params:
        input: []
        max_output_tokens: 8192
        temperature: 0.7
        top_p: 0.8

      aggregation_mode: mean
```

### NeMo-RL integration

To use this from an NRL GRPO config, add the config path and override
`judge_server_url`/`judge_model`:

```yaml
env:
  nemo_gym:
    config_paths:
      - resources_servers/multichallenge/configs/multichallenge.yaml
    multichallenge:
      resources_servers:
        multichallenge:
          judge_server_url: "0.0.0.0:8000"
          judge_model: "Qwen/Qwen3-30B-A3B-Instruct-2507"
          judge_responses_create_params:
            max_output_tokens: 8192
            temperature: ${policy.generation.temperature}
            top_p: ${policy.generation.top_p}
```

## Testing

```bash
cd 3rdparty/Gym-workspace/Gym && conda run -n trashrepo_ultra_v3 env PYTHONPATH=. \
  python -m pytest resources_servers/multichallenge/tests/ -v --timeout=60
```

Tests cover (no network required):
- config required-field enforcement (`judge_server_url` / `judge_model` / `judge_responses_create_params`)
- absence of the Gym-managed `judge_model_server` field
- `aggregation_mode` enum coercion (the inherited `verify()` reads `.value`)
- inherited score aggregation (mean / all)
- `[[YES]]` / `[[NO]]` verdict scanning on chat-completions judge text via the
  overridden `_evaluate_rubric_item`

Shared judge-transport tests live in
`resources_servers/utils_outsource/tests/test_judge_server_url_utils.py`.

## File Structure

```
multichallenge/
├── app.py                          # Server subclass + chat-completions judge
├── requirements.txt                # -e nemo-gym[dev] @ ../../
├── README.md                       # This file
├── configs/
│   └── multichallenge.yaml         # Server + agent configuration
├── data/
│   ├── example.jsonl               # Example data (committed)
│   ├── example_metrics.json        # Example metrics output (committed)
│   ├── example_rollouts.jsonl      # Example rollouts (committed)
│   ├── .gitignore
│   └── README.md
└── tests/
    ├── __init__.py
    └── test_multichallenge.py
```

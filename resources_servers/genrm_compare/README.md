# GenRM Compare Environment

Compares multiple candidate responses using a GenRM model via pairwise
comparisons. The GenRM model judge's hosting is **not managed by NeMo-Gym** —
the YAML config supplies a `genrm_server_url` (`host:port` of an
already-running `vllm serve` endpoint) and a `genrm_model` name. The GenRM
model is queried via the OpenAI **Chat Completions** API
(`{genrm_server_url}/v1/chat/completions`), which is the surface a stock
`vllm serve` exposes.

All pairwise comparison / aggregation / parse / cohort-buffering /
data-schema logic is inherited unchanged from
[`resources_servers/genrm_compare_original`](../genrm_compare_original), which
is the original Gym-managed-judge variant.

## How it differs from `genrm_compare_original`

| Aspect | `genrm_compare_original` | `genrm_compare` (this) |
|--------|--------------------------|------------------------|
| GenRM judge hosting | NeMo-Gym manages a `responses_api_models` vLLM server | Externally hosted; YAML supplies `genrm_server_url` |
| Judge protocol | `/v1/responses` (Responses API) via `ServerClient` | `/v1/chat/completions` (native vLLM) via direct HTTP |
| Pair payload | `response_1` / `response_2` / `principle` smuggled via `metadata` | `response_1` / `response_2` / `principle` passed as custom-role chat messages |
| Config fields | `genrm_model_server` (a `ModelServerRef`) | `genrm_server_url` + `genrm_model` (both mandated, no defaults) |
| Init validation | (none — Gym spins up the judge) | `genrm_server_url` normalized + `genrm_model` checked against `/v1/models` |
| Comparison / aggregation / parse / cohort verify / data schema | — | **Identical** (inherited from `genrm_compare_original`) |

Everything else — cohort-based `verify()` buffering
(`num_rollouts_per_prompt`), the `/compare` batch endpoint, pair generation
(`circular` / `all_pairs`), the `simple_tiebreaker` aggregation with length
bonuses, GenRM JSON-output parsing with `genrm_parse_retries`, and the
request/response models — is inherited unchanged.

## Data compatibility

Any dataset compatible with `genrm_simple_agent` (the original variant under
`genrm_compare_original/`) works here **unchanged** — the `agent_ref.name` is
the same (`genrm_simple_agent`). See `data/README.md` for details.

## Configuration

```yaml
genrm_compare_resources_server:
  resources_servers:
    genrm_compare:
      entrypoint: app.py

      # MANDATED (no defaults):
      genrm_server_url: "0.0.0.0:8000"   # host:port of the external vLLM GenRM judge
      genrm_model: "nvidia/Qwen3-Nemotron-235B-A22B-GenRM"

      num_rollouts_per_prompt: 16

      genrm_responses_create_params:
        input: []
        max_output_tokens: 16384
        temperature: 0.6
        top_p: 0.95

      comparison_strategy: circular
```

### NeMo-RL integration

To use this from an NRL GRPO config, add the config path and override
`genrm_server_url`/`genrm_model`:

```yaml
env:
  nemo_gym:
    config_paths:
      - resources_servers/genrm_compare/configs/genrm_compare.yaml
    genrm_compare_resources_server:
      resources_servers:
        genrm_compare:
          genrm_server_url: "0.0.0.0:8000"
          genrm_model: "nvidia/Qwen3-Nemotron-235B-A22B-GenRM"
```

## Testing

```bash
cd 3rdparty/Gym-workspace/Gym && conda run -n trashrepo_ultra_v3 env PYTHONPATH=. \
  python -m pytest resources_servers/genrm_compare/tests/ -v --timeout=60
```

Tests cover (no network required):
- config required-field enforcement (`genrm_server_url` / `genrm_model` / `genrm_responses_create_params`)
- absence of the Gym-managed `genrm_model_server` field
- outsource config defaults matching the `_original` config defaults
- inherited comparison endpoints (single-response default, verify default, pair generation)
- the overridden `_run_single_comparison` chat-completions transport: custom
  `response_1` / `response_2` / `principle` roles in the payload, the
  `max_retries=1, raise_on_context_length_error=True` transport flags, JSON
  verdict parsing, `GenRMOutputParseError` retries, and `BadRequestError`
  falling back to default scores

Shared judge-transport tests live in
`resources_servers/utils_outsource/tests/test_judge_server_url_utils.py`.

## File Structure

```
genrm_compare/
├── app.py                      # Server subclass + chat-completions GenRM judge
├── requirements.txt            # -e nemo-gym[dev] @ ../../
├── README.md                   # This file
├── configs/
│   └── genrm_compare.yaml      # Server + agent configuration
├── data/
│   ├── example.jsonl           # Example data (committed)
│   ├── example_metrics.json    # Example metrics output (committed)
│   ├── example_rollouts.jsonl  # Example rollouts (committed)
│   ├── .gitignore
│   └── README.md
└── tests/
    ├── __init__.py
    └── test_app.py
```

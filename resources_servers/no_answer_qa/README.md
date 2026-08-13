# No Answer QA Environment

Trajectory-collector QA environment: Q&A-style samples with **no expected
answers**. No judge is invoked, no reward models are called, and no verifiers
are run. The reward is **ALWAYS 0.0** regardless of the assistant output —
the environment simply returns the full trajectory to the training side
(with 0.0 reward, of course).

## How it works

The resources server overrides nothing but `verify()`: it echoes the request
back with `reward=0.0`. Every rollout is therefore collected verbatim and
fed to training with a constant zero reward, making this environment useful
for studying policy behavior on open-ended questions without any answer
signal.

Dataset rows may still carry leftover `general_qa` fields (`expected_answer`,
`question`, `should_use_judge`) when the data is reused from general_qa —
they are ignored entirely.

## How it differs from `general_qa`

| Aspect | `general_qa` | `no_answer_qa` (this) |
|--------|--------------|-----------------------|
| Expected answers | Required per sample | None |
| Deterministic verifiers | exact match + math_verify + F1 | None |
| LLM judge | Optional (external host:port) | None |
| Reward | 0.0–1.0 from verifiers/judge | Always 0.0 |
| Purpose | Reward QA responses | Collect trajectories |

## Configuration

```yaml
no_answer_qa:
  resources_servers:
    no_answer_qa:
      entrypoint: app.py
      domain: knowledge
      verified: false
```

Train/validation data reuses general_qa's HF/gitlab identifiers
(`nvidia/Nemotron-RL-knowledge-general-qa`, gitlab dataset `general_qa`
version `0.0.1`). Only `data/example.jsonl` (QA prompts, no answers) is
committed.

## Testing

```bash
cd 3rdparty/Gym-workspace/Gym && conda run -n trashrepo_ultra_v3 env PYTHONPATH=. \
  python -m pytest resources_servers/no_answer_qa/tests/ -v --timeout=60
```

Tests cover (no network required):
- `verify()` always returns reward `0.0` (text, empty, tool-call, multi-message outputs)
- The verify response echoes `responses_create_params` and `response`
- Leftover general_qa fields (`expected_answer`, `should_use_judge`, `question`) are ignored
- Config default name enforcement

## File Structure

```
no_answer_qa/
├── app.py                    # Minimal trajectory-collector server (always 0.0 reward)
├── requirements.txt          # -e nemo-gym[dev] @ ../../
├── README.md                 # This file
├── configs/
│   └── no_answer_qa.yaml     # Server + agent configuration
├── data/
│   ├── example.jsonl         # Example data (committed)
│   └── .gitignore
└── tests/
    ├── __init__.py
    └── test_app.py
```

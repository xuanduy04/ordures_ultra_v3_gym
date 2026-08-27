# General QA Environment

Evaluates model responses on general question-answering tasks using three
deterministic verifiers (exact match, math-verify, F1) and an optional LLM
judge whose hosting is **not managed by NeMo-Gym**. The YAML config supplies
a `judge_server_url` (`host:port` of an already-running `vllm serve` endpoint)
and a `judge_model` name. The judge is queried via the OpenAI
**Chat Completions** API (`{judge_server_url}/v1/chat/completions`).

## How it works

### Deterministic verification

Three verifiers are applied and the **best** score is taken:

| Verifier | Description |
|----------|-------------|
| `exact_match_verifier` | Case-insensitive, whitespace-normalized string comparison |
| `math_verify_verifier` | Hugging Face Math-Verify library: extracts and compares mathematical expressions (supports `\boxed{}` and raw numbers) |
| `F1_verifier` | Token-level F1 score over normalized text (punctuation stripped, stopwords removed) |

Answer extraction tries `\boxed{}` first, then multilingual `Answer:` markers as
fallback. If neither is found, the raw generated text is used.

### LLM judge (optional)

When `should_use_judge=true` **and** the deterministic reward is ≤ 0.5, the
external LLM judge is invoked. The judge uses an Arena-Hard-style **few-shot**
prompt (instructions plus eight worked equivalence examples in the system
message) asking whether the generated answer is equivalent to the expected
answer, using `[[A=B]]` / `[[A!=B]]` verdict labels. The judge's user turn
always includes the **question** field: the question is only used to search
for formatting/stylistic requests, never to judge answer correctness. To
eliminate positional bias, both answer orderings are evaluated.

### Dataset format

Each JSONL row must carry a top-level `question` field alongside
`expected_answer` (and optionally `should_use_judge`):

```json
{
  "agent_ref": {"name": "general_qa_simple_agent"},
  "responses_create_params": {"input": [{"role": "user", "content": "What is the capital of France?"}]},
  "question": "What is the capital of France?",
  "expected_answer": "Paris",
  "should_use_judge": true
}
```

If `question` is missing, non-string, or empty, the server substitutes a
"question not found" sentinel that instructs the judge to compare the answers
by content alone.

## How it differs from `math_with_judge`

| Aspect | `math_with_judge` | `general_qa` (this) |
|--------|-------------------|---------------------|
| Deterministic verifiers | `math_verify` library directly | Three-stage: exact match + math_verify + F1 |
| Answer extraction | Boxed + answer-colon via `utils_qa` | Same (shared `utils_qa` module) |
| Judge hosting | Gym-managed model server | Externally hosted (host:port) |
| Judge protocol | `/v1/responses` | `/v1/chat/completions` |
| Domain | math | knowledge |

## Configuration

```yaml
general_qa:
  resources_servers:
    general_qa:
      entrypoint: app.py

      # MANDATED (no defaults):
      judge_server_url: "0.0.0.0:8000"   # host:port of the external vLLM judge
      judge_model: "Qwen/Qwen3-30B-A3B-Instruct-2507"

      judge_responses_create_params:
        input: []
        max_output_tokens: 8192
        temperature: 0.7
        top_p: 0.8

      should_use_judge: true
```

## Testing

```bash
cd 3rdparty/Gym-workspace/Gym && conda run -n trashrepo_ultra_v3 env PYTHONPATH=. \
  python -m pytest resources_servers/general_qa/tests/ -v --timeout=60
```

Tests cover (no network required):
- `_build_judge_response` wrapping
- Config required-field enforcement (`judge_server_url` / `judge_model` / `judge_responses_create_params`)
- Deterministic verifiers (exact match, math_verify, F1)
- Answer extraction (boxed, answer-colon, empty)
- `_verify_answer_deterministically` end-to-end
- Judge user turn contains the `question` field (mocked chat-completions POST)
- Missing / non-string / empty `question` falls back to the "question not found" sentinel
- Verdict parsing (`[[A=B]]` / `[[A!=B]]`), swap pass, and judge skip when `should_use_judge=false`

## File Structure

```
general_qa/
├── app.py                    # Standalone server class + external judge integration
├── requirements.txt          # -e nemo-gym[dev] @ ../../, math-verify
├── README.md                 # This file
├── configs/
│   └── general_qa.yaml       # Server + agent configuration
├── data/
│   ├── example.jsonl         # Example data (committed)
│   └── .gitignore
└── tests/
    ├── __init__.py
    └── test_app.py
```

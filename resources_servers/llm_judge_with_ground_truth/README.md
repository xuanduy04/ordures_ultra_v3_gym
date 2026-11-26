# LLM Judge With Ground Truth Environment

Scores each rollout on its own with an externally-hosted chat model used as an
LLM judge (`llm_judge_server_url` + `llm_judge_model`, queried via
`/v1/chat/completions`). The judge gets a system prompt and a user prompt, so
any chat model works; GenRM models are not supported.

1. Preference (2-6): the judge compares the policy's answer with the sample's
   `ground_truth` twice, concurrently: once with the policy's answer as Answer A
   and once as Answer B. Each call outputs `[[A>B]]`, `[[B>A]]` or `[[A=B]]` and
   scores 3 if the policy's answer is preferred, 2 if neither, 1 if the ground
   truth is preferred. Preference is the sum of the two calls. If the two
   answers are equal up to whitespace, both calls are skipped and scored 2.
2. Helpfulness (1-5): the judge rates the policy's answer to the `question`,
   without seeing the ground truth.

```
reward = preference + helpfulness
```

The verdict is the last `[[...]]` in the judge's reply. A reply without a valid
verdict or rating is requested again up to `llm_judge_parse_retries` times, then
scored 2 (that preference call) or 3 (helpfulness). Begin/end markers
(`<|begin_of_…|>`, `<|end_of_…|>`) are stripped from the answers before they go
into either prompt. An empty policy answer gets reward 3 without calling the
judge.

The judge prompts are class attributes in `app.py` and can only be changed
there.

## Metrics

Besides `reward`, each verify response carries `preference_score`,
`helpfulness_score`, `win_rates` (the share of the two preference verdicts that
prefer the policy's answer) and `win_equal_rates` (the share that prefer it or
call a tie; a missing verdict counts as neither, and an answer identical to the
ground truth counts as two ties). NeMo RL logs every numeric field of the response per agent,
e.g. `train/llm_judge_with_ground_truth_simple_agent/win_rates/mean`.

## Data

Each row must have `question` (the user prompt shown to the judge) and
`ground_truth` strings:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "What is 12 * 7?"}]}, "question": "What is 12 * 7?", "ground_truth": "12 * 7 = 84.", "agent_ref": {"type": "responses_api_agents", "name": "llm_judge_with_ground_truth_simple_agent"}}
```

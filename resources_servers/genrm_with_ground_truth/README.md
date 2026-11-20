# GenRM With Ground Truth Environment

Scores each rollout on its own with an externally-hosted GenRM model, using the
same endpoint setup as [`genrm_compare`](../genrm_compare) (`genrm_server_url` +
`genrm_model`, queried via `/v1/chat/completions`).

- `response_1` = the sample's `ground_truth`
- `response_2` = the policy's answer

```
reward = score_2 + ranking
```

`score_2` (1-5) is the GenRM helpfulness score of the policy's answer; `ranking`
(1-6) is 1 when the ground truth is much better and 6 when the policy's answer
is much better. The reward range is [2, 11]. On GenRM failure the reward is
`default_score + default_ranking`.

## Data

Each row must have a `ground_truth` string:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "What is 12 * 7?"}]}, "ground_truth": "12 * 7 = 84.", "agent_ref": {"type": "responses_api_agents", "name": "genrm_with_ground_truth_simple_agent"}}
```

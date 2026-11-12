# Math With Tool And Judge Environment

## Details

**Environment**. A math agent can call the `calculate` tool while solving a
problem, read the result, and then produce its final answer. The resource server
serves `/calculate` and `/verify` in the same FastAPI application.

**Domain**. Math and arithmetic tool use.

**Implementation**. `app.py` imports the evaluator from `calculator.py` within
this environment and inherits answer verification from
[`math_with_judge`](../math_with_judge/README.md). Calculator requests are
stateless. Each expression contains all inputs needed for that calculation.

The rollout flow is sequential.

1. The policy model receives the question and the `calculate` function schema.
2. The agent routes a `calculate` function call to `POST /calculate`.
3. The tool result is appended to the conversation and returned to the policy.
4. The policy produces its final answer, conventionally inside `\boxed{}`.
5. The agent submits the response, question, and reference answer to `/verify`.

## Dependencies

The environment uses the dependencies already required by `math_with_judge`.
The calculator evaluator uses Python's `ast`, `operator`, and `math` modules.

For an existing Gym checkout with its `.venv` and Gym dependencies installed,
install the verifier dependency from the repository root.

```bash
UV_NO_CONFIG=1 uv pip install --python .venv/bin/python 'math-verify==0.8.0'
```

The commands below use `.venv/bin/python` directly so project synchronization
does not remove environment-specific dependencies.

## Supported Calculations

| Operation | Expression | Result |
| --- | --- | --- |
| Addition and subtraction | `5 + 3 - 2` | `6` |
| Multiplication | `6 * 7` | `42` |
| Division | `7 / 2` | `3.5` |
| Floor division | `-7 // 2` | `-4` |
| Modulo | `7 % 3` | `1` |
| Power | `2 ** 10` | `1024` |
| Negative exponent | `2 ** -3` | `0.125` |
| Negative base | `(-2) ** 3` | `-8` |
| Fractional exponent | `9 ** 0.5` | `3.0` |
| Parentheses and unary signs | `-(2 + 3) * 4` | `-20` |
| Exponential | `exp(1)` | Approximately `2.718281828459045` |
| Natural logarithm | `log(e)` | `1.0` |
| Logarithm with an explicit base | `log(8, 2)` | `3.0` |

Numeric inputs include integers, decimals, and scientific notation such as
`1e-3`. The available constants are `pi`, `e`, and `tau`. Functions can be
nested, for example `exp(log(5))`.

Python operator precedence applies. `(-2) ** 2` evaluates to `4`, while
`-2 ** 2` evaluates to `-4`. Floating-point calculations follow Python's
numeric behavior. Results must be finite real numbers. A negative base with a
fractional exponent can produce a complex result and cause an error.

`log(x)` requires `x > 0`. For `log(x, base)`, the base must also be positive
and different from `1`. The maximum expression length is 1,000 characters.

## Run Locally

Run the following command from the Gym repository root. It starts the integrated
resource server on `127.0.0.1:8003` with library verification enabled and the
LLM judge disabled. An empty `ServerClient` configuration supports these local
calculator and library-verification calls.

```bash
.venv/bin/python -c '
import uvicorn
from omegaconf import OmegaConf

from nemo_gym.server_utils import ServerClient
from resources_servers.math_with_tool_and_judge.app import (
    MathWithToolAndJudgeConfig,
    MathWithToolAndJudgeServer,
)

config = MathWithToolAndJudgeConfig(
    host="127.0.0.1",
    port=8003,
    entrypoint="app.py",
    name="math_with_tool_and_judge",
    should_use_judge=False,
    judge_server_url="",
    judge_model="",
    judge_responses_create_params={"input": []},
)
client = ServerClient(
    head_server_config={"host": "127.0.0.1", "port": 11000},
    global_config_dict=OmegaConf.create({}),
)
server = MathWithToolAndJudgeServer(config=config, server_client=client)
uvicorn.run(
    server.setup_webserver(),
    host=config.host,
    port=config.port,
    access_log=False,
)
'
```

Keep this terminal running while issuing requests from another terminal.
The integrated application exposes its generated API schema at `/docs`.

## Calculator API

```bash
curl -sS http://127.0.0.1:8003/calculate \
  -H 'Content-Type: application/json' \
  -d '{"expression":"log(8, 2) + 2 ** -3"}'
```

```json
{"expression":"log(8, 2) + 2 ** -3","result":3.125}
```

Successful requests return HTTP 200. Evaluation errors such as `1 / 0` and
`log(-1)` return HTTP 400 with an `error` field. Invalid request schemas,
including a missing expression or an expression longer than 1,000 characters,
return FastAPI's HTTP 422 validation response.

Register the following function in the policy's Responses API request. The
Gym `simple_agent` routes its function name to `/calculate` on the configured
resource server.

```json
{
  "type": "function",
  "name": "calculate",
  "description": "Evaluate arithmetic expressions with exp, log, pi, e, and tau. Use ** for powers and log(x, base) for a logarithm with a specified base.",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": "An arithmetic expression of at most 1000 characters."
      }
    },
    "required": ["expression"],
    "additionalProperties": false
  },
  "strict": true
}
```

## Answer Verification

`POST /verify` takes `question`, `expected_answer`,
`responses_create_params`, and the policy's Responses API `response` object.
The verifier reads assistant text from the response and extracts the answer
using the inherited math-verification logic.

```bash
curl -sS http://127.0.0.1:8003/verify \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is 6 * 7?",
    "expected_answer": "42",
    "responses_create_params": {"input": []},
    "response": {
      "id": "manual-response",
      "created_at": 0,
      "model": "manual-test",
      "object": "response",
      "parallel_tool_calls": false,
      "tool_choice": "none",
      "tools": [],
      "output": [{
        "id": "manual-message",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{
          "type": "output_text",
          "text": "\\boxed{42}",
          "annotations": []
        }]
      }]
    }
  }'
```

For this example, the response contains `reward=1.0`, `library_reward=1.0`,
and `judge_evaluations=null`. The full response also includes the submitted
request and extracted answer.

The inherited mixed-reward strategy runs `math_verify` first. If
`should_use_judge=True` and `library_reward <= 0.5`, the server calls the LLM
judge to compare the reference and generated answers. It checks both answer
orders when the first comparison returns equivalent.

To enable the external judge, supply an accessible `judge_server_url`, its
`judge_model`, `judge_responses_create_params`, and `should_use_judge=True`.
Startup validates the model through `/v1/models`. Judge requests use
`/v1/chat/completions`. The inherited library-verifier defaults are a
10-second timeout and a maximum concurrency of 32 verifier processes.

Full rollout collection additionally requires an agent configuration, a policy
model endpoint, and dataset rows containing the question, reference answer,
and the `calculate` tool schema in `responses_create_params.tools`.

## Performance

The integrated calculator handles requests through FastAPI and Uvicorn with
access logging disabled in the local command above. Throughput for this
integrated server remains to be measured. Calculator throughput and complete
rollout throughput should be measured separately because rollouts also include
policy generation and answer verification.

## File Structure

```text
math_with_tool_and_judge/
├── app.py          # Gym resource server with /calculate and inherited /verify
├── calculator.py   # Standard-library evaluator and standalone HTTP entrypoint
└── README.md       # Usage, tool schema, and verification flow
```

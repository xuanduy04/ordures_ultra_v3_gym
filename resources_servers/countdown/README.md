# Countdown Resources Server

### Overview
Rule-based verifier for the Countdown arithmetic task: combine every given number exactly once with `+ - * /` and
parentheses to reach a target. No judge model and no reference solution are used.

- Dataset: https://huggingface.co/datasets/yzhuang/tinyzero-Countdown-Tasks-3to4 (columns `nums`, `target`)
- Ported from TinyZero (Apache-2.0): https://github.com/Jiayi-Pan/TinyZero
  - Reward: https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py
  - Preprocessor: https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py

### Input schema
- `responses_create_params`: OpenAI Responses create params (single user turn, optional system prompt), e.g.
  `Using the numbers [44, 19, 35], create an equation that equals 98. ... Return the final answer in \boxed{}, for example \boxed{(1 + 2) / 3}.`
- `nums` (required): list of ints the equation must use, each exactly once.
- `target` (required): int the equation must evaluate to.

### Config
All required (no Python defaults):
- `format_score`: reward for an extracted but wrong equation (TinyZero: `0.1`).
- `score`: reward for a correct equation (TinyZero: `1.0`).
- `strict_ops`: reject `**` and `//`. `false` reproduces TinyZero exactly.

### Extraction (the one intentional difference from TinyZero)
TinyZero splits on `Assistant:` / `<|im_start|>assistant`, keeps only the **last line**, and reads the last
`<answer>...</answer>` there, so e.g. a trailing newline after `</answer>` scores 0. Here the verifier reads only the
assistant's `output_text` (never the prompt or reasoning items) and takes the content of the **last `\boxed{...}`**
(brace-matched, may span lines). `<answer>` tags and `Answer:` lines are ignored. No box, or an unclosed last box, scores 0.

### Scoring (same as TinyZero's `compute_score`)
1. No extracted equation: `0.0`.
2. Numbers in the equation (`\d+`) differ from `nums` as a multiset: `format_score`.
3. Characters outside `[\d+\-*/().\s]`: `format_score`.
4. `strict_ops` and the equation contains `**` or `//`: `format_score`.
5. `eval` with no builtins; error: `format_score`; `|result - target| < 1e-5`: `score`; else `format_score`.

`tests/tinyzero_countdown.py` is TinyZero's reward file vendored unchanged for the parity fuzz test.

# Licensing information
Code: Apache 2.0
Data: see the dataset card

Dependencies
- nemo_gym: Apache 2.0

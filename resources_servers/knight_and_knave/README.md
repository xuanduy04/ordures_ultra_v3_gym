# Knight and Knave Resources Server

### Overview
Rule-based verifier for Knights and Knaves logic puzzles: knights always tell the truth, knaves always lie, and the
model must name the role of every inhabitant. No judge model is used.

- Dataset format: https://huggingface.co/datasets/K-and-K/knights-and-knaves (column `solution_text_format`)
- Reward ported from: https://github.com/Leey21/A-Data-Centric-Study/blob/main/verl/utils/reward_score/kk.py

### Input schema
- `responses_create_params`: OpenAI Responses create params (single user turn), e.g.
  `<quiz>\nList the identity of each person one by one in \boxed{}, for example, for example, \boxed{(1) Alpha is a knight (2) Beta is a knave (3)...}`
- `solution_dictionary` (required): reference role of every inhabitant, e.g. `{"Zoey": "knave", "Oliver": "knight"}`;
  values must be `knight` or `knave`. It is precomputed offline from the dataset's `solution_text_format`
  (`(1) Zoey is a knave\n(2) Oliver is a knight`) with `parse_solution_text_format` from `app.py`, so nothing is parsed
  from the reference at verify time. `custom_utils/utils_data/custom_dataset_processors/knights-and-knaves/_processor.py`
  in the parent repo does this and cross-checks the result against the `names` / `solution` columns.

### Config
All required (no Python defaults):
- `incorrect_score`: reward for a `\boxed{}` answer that is unparseable or wrong (reference: `0`).
- `score`: reward for a fully correct answer (reference: `1.0`).

### Extraction (the one intentional difference from the reference)
The reference splits on `Assistant:` / `<|im_start|>assistant` and reads the last `<answer>...</answer>`. Here the
verifier reads only the assistant's `output_text` (never the prompt or reasoning items) and takes the content of the
**last `\boxed{...}`** (brace-matched, may span lines). `<answer>` tags are ignored. No box, an unclosed last box, or an
empty box scores `0.0`.

### Scoring (same as the reference `compute_score`)
1. Reference roles (offline, when building `solution_dictionary`): each line of `solution_text_format` maps its first
   alphabetic word to the first `knight`/`knave` after it; unmatched lines are skipped.
2. The boxed answer must contain `knight`/`knave` (case-insensitive substring count, so `knights` counts too)
   exactly once per inhabitant, else `incorrect_score`.
3. Every inhabitant must appear as `<Name> is a knight` / `<Name> is a knave` (whole-word name, case-insensitive;
   first occurrence wins), else `incorrect_score`.
4. All roles match the reference: `score`; otherwise `incorrect_score`.

The reference's `<think>`/`<answer>` structure check is disabled there and is not ported.

# Licensing information
Code: Apache 2.0
Data: `data/example.jsonl` is hand-written; see the dataset card for the full dataset.

Dependencies
- nemo_gym: Apache 2.0

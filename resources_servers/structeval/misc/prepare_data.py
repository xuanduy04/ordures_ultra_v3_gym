
#
# Converts StructEval dataset (TIGER-Lab/StructEval, MIT License) to Gym JSONL format.
"""Convert StructEval JSON dataset to Gym JSONL format."""

import argparse
import json


INSTRUCTION_SUFFIX = (
    "\n\nIMPORTANT: Only output the required output format. "
    "You must start the format/code with <|BEGIN_CODE|> and end the format/code with <|END_CODE|>. "
    "No other text output (explanation, comments, etc.) are allowed. "
    "Do not use markdown code fences."
)


def convert_record(record: dict) -> dict:
    """Convert a single StructEval record to Gym format."""
    query = record["query"] + INSTRUCTION_SUFFIX
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": query}],
        },
        "task_id": record["task_id"],
        "task_name": record["task_name"],
        "input_type": record["input_type"],
        "output_type": record["output_type"],
        "raw_output_metric": record["raw_output_metric"],
        "rendering": record.get("rendering", False),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert StructEval dataset to Gym JSONL")
    parser.add_argument("--input", required=True, help="Path to StructEval JSON dataset")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--example-output", default=None, help="Optional example JSONL path (subset)")
    parser.add_argument("--example-count", type=int, default=5, help="Number of example records")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for record in data:
            gym_record = convert_record(record)
            out.write(json.dumps(gym_record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} records to {args.output}")

    if args.example_output:
        # Pick one record per output_type for diversity.
        seen_types: set = set()
        examples: list = []
        for record in data:
            ot = record["output_type"]
            if ot not in seen_types and len(examples) < args.example_count:
                examples.append(convert_record(record))
                seen_types.add(ot)
        # Fill remaining slots if fewer output types than example_count.
        for record in data:
            if len(examples) >= args.example_count:
                break
            gym_rec = convert_record(record)
            if gym_rec not in examples:
                examples.append(gym_rec)

        with open(args.example_output, "w", encoding="utf-8") as out:
            for rec in examples:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(examples)} examples to {args.example_output}")


if __name__ == "__main__":
    main()

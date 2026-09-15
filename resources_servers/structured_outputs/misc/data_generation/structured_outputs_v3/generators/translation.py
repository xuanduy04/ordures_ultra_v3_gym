
"""Category 2: Format translation problems."""

import random
from typing import Any, Dict, List

from templates import (
    ALL_FORMATS,
    FORMAT_NAMES,
    TRANSLATION_TEMPLATES,
    make_gym_record,
    represent_schema,
)


def generate_translation(
    records: List[Dict[str, Any]],
    rng: random.Random,
    samples_per_record: int = 3,
    target_formats: List[str] = ALL_FORMATS,
    max_samples: int = 1000,
) -> List[Dict[str, Any]]:
    results = []
    for record in records:
        schema_dict = record["_json_schema"]
        rid = record.get("_record_id", "unknown")
        source_fmt = record.get("target_output_format", "json")
        source_output = record.get("target_output", "")

        if not source_output:
            continue

        other_formats = [f for f in target_formats if f != source_fmt]
        if not other_formats:
            continue

        for _ in range(samples_per_record):
            if len(results) >= max_samples:
                return results

            target_fmt = rng.choice(other_formats)
            schema_str = represent_schema(schema_dict, "json")

            prompt = rng.choice(TRANSLATION_TEMPLATES).format(
                source_format=FORMAT_NAMES.get(source_fmt, source_fmt),
                target_format=FORMAT_NAMES.get(target_fmt, target_fmt),
                source_output=source_output,
                schema=schema_str,
            )
            input_msgs = [{"role": "user", "content": prompt}]

            results.append(
                make_gym_record(
                    input_msgs=input_msgs,
                    schema_dict=schema_dict,
                    schema_type=target_fmt,
                    problem_type="translation",
                    schema_repr="json",
                    source_record_id=rid,
                    source_format=source_fmt,
                )
            )
    return results

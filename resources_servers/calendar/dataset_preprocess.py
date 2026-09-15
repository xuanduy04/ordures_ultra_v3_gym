
import json
from argparse import ArgumentParser

from tqdm import tqdm

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
)


def main(args):
    # Load the dataset
    with open(args.input, "r") as f:
        dataset = json.load(f)

    if args.exclude_success:
        dataset = [d for d in dataset if d["grade"] == 0]

    ds_out = []

    for sample in tqdm(dataset, desc=f"Processing samples from {args.input}", total=len(dataset)):
        conversation = sample["conversation"][:-1]  # remove the last message
        for m in conversation:
            if m["role"] == "assistant" and "reasoning_content" in m["content"]:
                del m["reasoning_content"]
        exp_cal_state = sample["exp_cal_state"]
        if len(exp_cal_state) == 0:
            continue
        response_create_params = NeMoGymResponseCreateParamsNonStreaming(input=conversation, tools=[])
        ds_out.append(
            json.dumps(
                {
                    "responses_create_params": response_create_params.model_dump(exclude_unset=True),
                    "exp_cal_state": exp_cal_state,
                }
            )
            + "\n"
        )

    ds_out_train = ds_out[: -args.n_val]
    ds_out_val = ds_out[-args.n_val :]

    with open(args.output_train, "w") as f:
        f.writelines(ds_out_train)
    with open(args.output_val, "w") as f:
        f.writelines(ds_out_val)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, help="Input JSONL file", default="./data/rollouts.json")
    parser.add_argument("--n_val", type=int, help="Number of validation samples", default=128)
    parser.add_argument(
        "--exclude_success",
        action="store_true",
        help="Excludes successful rollouts from the input file",
        default=False,
    )
    parser.add_argument("--output_train", type=str, help="Output JSONL file", default="./data/train.jsonl")
    parser.add_argument("--output_val", type=str, help="Output JSONL file", default="./data/validation.jsonl")
    args = parser.parse_args()
    main(args)


from benchmarks.ruler.prepare_utils import prepare_helper


def prepare():
    return prepare_helper(
        model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",  # pragma: allowlist secret
        length=262144,
        output_name="ruler_nemotron_3_256k_benchmark.jsonl",
    )


if __name__ == "__main__":
    prepare()

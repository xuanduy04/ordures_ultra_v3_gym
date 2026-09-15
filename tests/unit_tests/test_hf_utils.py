
from nemo_gym.hf_utils import HfApi


class TestHFUtils:
    def test_sanity(self) -> None:
        HfApi()

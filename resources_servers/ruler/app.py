

from typing import List, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class RulerResourcesServerConfig(BaseResourcesServerConfig):
    pass


class RulerVerifyRequest(BaseVerifyRequest):
    outputs: List[str]
    length: int
    subset: str


class RulerVerifyResponse(RulerVerifyRequest, BaseVerifyResponse):
    cwe: Optional[float] = None
    fwe: Optional[float] = None
    niah_multikey_1: Optional[float] = None
    niah_multikey_2: Optional[float] = None
    niah_multikey_3: Optional[float] = None
    niah_multiquery: Optional[float] = None
    niah_multivalue: Optional[float] = None
    niah_single_1: Optional[float] = None
    niah_single_2: Optional[float] = None
    niah_single_3: Optional[float] = None
    qa_1: Optional[float] = None
    qa_2: Optional[float] = None
    vt: Optional[float] = None


class RulerResourcesServer(SimpleResourcesServer):
    config: RulerResourcesServerConfig

    async def verify(self, body: RulerVerifyRequest) -> BaseVerifyResponse:
        prediction = body.response.output_text.strip()

        if body.subset in ("qa_1", "qa_2"):
            reward_calc_function = self.string_match_part_single
        else:
            reward_calc_function = self.string_match_all_single
        reward = reward_calc_function(prediction, body.outputs)

        return RulerVerifyResponse(
            **body.model_dump(),
            reward=reward,
            **{body.subset: reward},
        )

    # These helper functions are taken from https://github.com/NVIDIA-NeMo/Skills/blob/54d2e113c2f64bf74bda72e15f23f01b524850da/nemo_skills/evaluation/evaluator/ruler.py#L36
    def string_match_all_single(self, pred: str, refs: List[str]) -> float:
        return sum([1.0 if r.lower() in pred.lower() else 0.0 for r in refs]) / len(refs)

    def string_match_part_single(self, pred: str, refs: List[str]) -> float:
        return max([1.0 if r.lower() in pred.lower() else 0.0 for r in refs])


if __name__ == "__main__":
    RulerResourcesServer.run_webserver()

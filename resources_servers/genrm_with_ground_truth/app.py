"""
GenRM With Ground Truth Resources Server.

Scores each rollout on its own with an externally-hosted GenRM model (see
:mod:`resources_servers.genrm_compare`). The sample's mandated ``ground_truth``
is sent as ``response_1`` and the policy's answer as ``response_2``.

reward = score_2 + ranking, where score_2 (1-5) is the helpfulness of the
policy's answer and ranking (1-6) is higher when the policy's answer is
preferred over the ground truth.
"""

from __future__ import annotations

from nemo_gym.base_resources_server import BaseVerifyResponse
from resources_servers.genrm_compare.app import GenRMCompareConfig, GenRMCompareResourcesServer
from resources_servers.genrm_compare_original.app import GenRMCompareVerifyRequest, _input_to_conversation_history


class GenRMWithGroundTruthConfig(GenRMCompareConfig):
    name: str = "genrm_with_ground_truth"


class GenRMWithGroundTruthVerifyRequest(GenRMCompareVerifyRequest):
    ground_truth: str


class GenRMWithGroundTruthResourcesServer(GenRMCompareResourcesServer):
    config: GenRMWithGroundTruthConfig

    async def verify(self, body: GenRMWithGroundTruthVerifyRequest) -> BaseVerifyResponse:
        ground_truth_obj = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": body.ground_truth}]}]
        }
        _, score_2, ranking = await self._run_single_comparison(
            _input_to_conversation_history(body.responses_create_params.input),
            ground_truth_obj,
            body.response.model_dump(),
            principle=body.principle,
        )
        return BaseVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=score_2 + ranking,
        )


if __name__ == "__main__":
    GenRMWithGroundTruthResourcesServer.run_webserver()

"""
No Answer QA Environment Resources Server.

This is a trajectory-collector QA resource server: Q&A-style samples with
**no expected answers**. No judge is invoked, no reward models are called,
and no verifiers are run. The reward is ALWAYS 0.0 regardless of the
assistant output, so the environment simply returns the full trajectory to
the training side (with 0.0 reward).

Dataset rows may still carry leftover general_qa fields (``expected_answer``,
``question``, ``should_use_judge``) when the data is reused from general_qa —
they are ignored entirely.
"""

from __future__ import annotations

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class NoAnswerQAResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the No Answer QA environment server."""

    name: str = "no_answer_qa"


class NoAnswerQAResourcesServer(SimpleResourcesServer):
    """Resources server that always rewards 0.0, acting as a trajectory collector."""

    config: NoAnswerQAResourcesServerConfig

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Return the trajectory with a constant 0.0 reward.

        Args:
            body: The verify request containing the rollout's
                ``responses_create_params`` and the model ``response``.

        Returns:
            A verify response echoing the request with ``reward=0.0``.
        """
        return BaseVerifyResponse(**body.model_dump(), reward=0.0)


if __name__ == "__main__":
    NoAnswerQAResourcesServer.run_webserver()

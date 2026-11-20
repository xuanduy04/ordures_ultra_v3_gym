"""Tests for the GenRM With Ground Truth Resources Server."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.genrm_with_ground_truth.app import (
    GenRMWithGroundTruthConfig,
    GenRMWithGroundTruthResourcesServer,
    GenRMWithGroundTruthVerifyRequest,
)


def _make_server():
    cfg = GenRMWithGroundTruthConfig(
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="rlhf",
        genrm_server_url="0.0.0.0:8000",
        genrm_model="test-genrm-model",
        genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=1024),
    )
    return GenRMWithGroundTruthResourcesServer.model_construct(config=cfg, server_client=MagicMock())


def _make_request(**extra):
    return GenRMWithGroundTruthVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "What is 2+2?"}]
        ),
        response=NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text="Four", type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ),
        **extra,
    )


def _genrm_response(score_1, score_2, ranking):
    content = json.dumps({"score_1": score_1, "score_2": score_2, "ranking": ranking})
    return {"choices": [{"message": {"content": content}}]}


def test_reward_is_score_2_plus_ranking(monkeypatch) -> None:
    server = _make_server()
    post_mock = AsyncMock(return_value=_genrm_response(5, 2, 4))
    monkeypatch.setattr("resources_servers.genrm_compare.app._post_chat_completions", post_mock)

    res = asyncio.run(server.verify(_make_request(ground_truth="4")))

    assert res.reward == pytest.approx(6.0)
    messages = post_mock.await_args.args[2]["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "What is 2+2?"),
        ("response_1", "4"),
        ("response_2", "Four"),
    ]


def test_missing_ground_truth_raises() -> None:
    with pytest.raises(ValidationError):
        _make_request()

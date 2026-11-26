"""Tests for the LLM Judge With Ground Truth Resources Server."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.llm_judge_with_ground_truth.app import (
    LLMJudgeWithGroundTruthConfig,
    LLMJudgeWithGroundTruthResourcesServer,
    LLMJudgeWithGroundTruthVerifyRequest,
)


QUESTION = "What is 2+2?"


def _make_server():
    cfg = LLMJudgeWithGroundTruthConfig(
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="rlhf",
        llm_judge_server_url="0.0.0.0:8000",
        llm_judge_model="test-judge-model",
        llm_judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=1024),
    )
    return LLMJudgeWithGroundTruthResourcesServer.model_construct(config=cfg, server_client=MagicMock())


def _make_request(answer="Four", **extra):
    return LLMJudgeWithGroundTruthVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": QUESTION}]),
        response=NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text=answer, type="output_text")],
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


def _mock_judge(monkeypatch, policy_first, policy_second, helpfulness):
    server = LLMJudgeWithGroundTruthResourcesServer
    second_prompt = server.PREFERENCE_PROMPT_TEMPLATE.format(question=QUESTION, answer_a="4", answer_b="Four")

    async def post(env_name, url, payload, **kwargs):
        system, user = (message["content"] for message in payload["messages"])
        if system != server.PREFERENCE_SYSTEM_MESSAGE:
            replies = helpfulness
        else:
            replies = policy_second if user == second_prompt else policy_first
        return {"choices": [{"message": {"content": replies.pop(0)}}]}

    post_mock = AsyncMock(side_effect=post)
    monkeypatch.setattr("resources_servers.llm_judge_with_ground_truth.app._post_chat_completions", post_mock)
    return post_mock


def _sent_messages(post_mock):
    return [call.args[2]["messages"] for call in post_mock.await_args_list]


def test_both_orders_and_helpfulness_are_judged(monkeypatch) -> None:
    server = _make_server()
    post_mock = _mock_judge(monkeypatch, ["Verdict: [[A>B]]"], ["Verdict: [[B>A]]"], ["Helpfulness: [[4]]"])

    res = asyncio.run(server.verify(_make_request(question=QUESTION, ground_truth="4")))

    assert res.reward == pytest.approx(10.0)
    assert (res.preference_score, res.helpfulness_score, res.win_rates, res.win_equal_rates) == (6.0, 4.0, 1.0, 1.0)
    assert post_mock.await_count == 3
    assert {tuple((m["role"], m["content"]) for m in messages) for messages in _sent_messages(post_mock)} == {
        (
            ("system", server.PREFERENCE_SYSTEM_MESSAGE),
            ("user", server.PREFERENCE_PROMPT_TEMPLATE.format(question=QUESTION, answer_a="Four", answer_b="4")),
        ),
        (
            ("system", server.PREFERENCE_SYSTEM_MESSAGE),
            ("user", server.PREFERENCE_PROMPT_TEMPLATE.format(question=QUESTION, answer_a="4", answer_b="Four")),
        ),
        (
            ("system", server.HELPFULNESS_SYSTEM_MESSAGE),
            ("user", server.HELPFULNESS_PROMPT_TEMPLATE.format(question=QUESTION, answer="Four")),
        ),
    }


def test_verdicts_are_scored_from_the_policy_side(monkeypatch) -> None:
    server = _make_server()
    _mock_judge(monkeypatch, ["[[B>A]]"], ["[[A=B]]"], ["Helpfulness: [[2]]"])

    res = asyncio.run(server.verify(_make_request(question=QUESTION, ground_truth="4")))

    assert res.reward == pytest.approx(5.0)
    assert (res.preference_score, res.helpfulness_score, res.win_rates, res.win_equal_rates) == (3.0, 2.0, 0.0, 0.5)


def test_invalid_replies_fall_back_after_retries(monkeypatch) -> None:
    server = _make_server()
    attempts = 1 + server.config.llm_judge_parse_retries
    post_mock = _mock_judge(monkeypatch, ["No verdict."] * attempts, ["[[A>B]]"], ["Rating: [[9]]"] * attempts)

    res = asyncio.run(server.verify(_make_request(question=QUESTION, ground_truth="4")))

    assert res.reward == pytest.approx(6.0)
    assert (res.preference_score, res.helpfulness_score, res.win_rates, res.win_equal_rates) == (3.0, 3.0, 0.0, 0.0)
    assert post_mock.await_count == 2 * attempts + 1


def test_identical_answer_skips_the_pairwise_judge(monkeypatch) -> None:
    server = _make_server()
    post_mock = _mock_judge(monkeypatch, [], [], ["Helpfulness: [[3]]"])

    res = asyncio.run(server.verify(_make_request(answer="4 ", question=QUESTION, ground_truth=" 4")))

    assert res.reward == pytest.approx(7.0)
    assert (res.preference_score, res.helpfulness_score, res.win_rates, res.win_equal_rates) == (4.0, 3.0, 0.0, 1.0)
    assert post_mock.await_count == 1


def test_layout_markers_are_stripped_from_answers(monkeypatch) -> None:
    server = _make_server()
    post_mock = _mock_judge(monkeypatch, ["[[A>B]]"], ["[[B>A]]"], ["Helpfulness: [[5]]"])

    asyncio.run(
        server.verify(
            _make_request(answer="Four<|end_of_answer|><|begin_of_answer_B|>", question=QUESTION, ground_truth="4")
        )
    )

    assert {messages[1]["content"] for messages in _sent_messages(post_mock)} == {
        server.PREFERENCE_PROMPT_TEMPLATE.format(question=QUESTION, answer_a="Four", answer_b="4"),
        server.PREFERENCE_PROMPT_TEMPLATE.format(question=QUESTION, answer_a="4", answer_b="Four"),
        server.HELPFULNESS_PROMPT_TEMPLATE.format(question=QUESTION, answer="Four"),
    }


def test_empty_answer_gets_minimum_reward_without_judge_calls(monkeypatch) -> None:
    server = _make_server()
    post_mock = _mock_judge(monkeypatch, [], [], [])

    res = asyncio.run(server.verify(_make_request(answer=" ", question=QUESTION, ground_truth="4")))

    assert res.reward == pytest.approx(3.0)
    assert (res.preference_score, res.helpfulness_score, res.win_rates, res.win_equal_rates) == (2.0, 1.0, 0.0, 0.0)
    post_mock.assert_not_awaited()


@pytest.mark.parametrize("extra", [{"question": QUESTION}, {"ground_truth": "4"}])
def test_missing_question_or_ground_truth_raises(extra) -> None:
    with pytest.raises(ValidationError):
        _make_request(**extra)

"""Tests for the outsourced GenRM Compare Resources Server."""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import BadRequestError

from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient

from resources_servers.genrm_compare.app import (
    GenRMCompareConfig,
    GenRMCompareRequest,
    GenRMCompareResourcesServer,
)
from resources_servers.genrm_compare_original.app import GenRMCompareVerifyRequest


def _make_config(**overrides):
    base = dict(
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="rlhf",
        name="test",
        genrm_server_url="0.0.0.0:8000",
        genrm_model="test-genrm-model",
        genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=1024
        ),
    )
    base.update(overrides)
    return GenRMCompareConfig(**base)


class TestConfigRequiredFields:
    def test_missing_genrm_server_url_raises(self):
        with pytest.raises(Exception):
            _make_config(genrm_server_url=None)

    def test_missing_genrm_model_raises(self):
        with pytest.raises(Exception):
            _make_config(genrm_model=None)

    def test_missing_genrm_responses_create_params_raises(self):
        with pytest.raises(Exception):
            _make_config(genrm_responses_create_params=None)

    def test_valid_config(self):
        cfg = _make_config()
        assert cfg.genrm_server_url == "0.0.0.0:8000"
        assert cfg.genrm_model == "test-genrm-model"

    def test_default_name(self):
        cfg = GenRMCompareConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            domain="rlhf",
            genrm_server_url="0.0.0.0:8000",
            genrm_model="m",
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        assert cfg.name == "genrm_compare"

    def test_no_genrm_model_server_field(self):
        # This config must not carry the Gym-managed genrm_model_server.
        cfg = _make_config()
        assert not hasattr(cfg, "genrm_model_server") or "genrm_model_server" not in cfg.model_fields


class TestInheritedComparison:
    """Comparison logic is inherited unchanged from the original GenRMCompareResourcesServer."""

    def _make_response_obj(self, output_text: str) -> Dict[str, Any]:
        return {
            "id": "resp_123",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": output_text}],
                }
            ],
        }

    async def test_compare_single_response_returns_default(self) -> None:
        """Single response returns default score (no comparison possible)."""
        cfg = _make_config()
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer.model_construct(config=cfg, server_client=server_mock)

        req = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}],
            response_objs=[self._make_response_obj("Response 1")],
        )

        res = await rs.compare(req)

        assert len(res.rewards) == 1
        assert res.rewards[0] == pytest.approx(3.0)
        server_mock.post.assert_not_called()

    async def test_verify_returns_default(self) -> None:
        """Verify returns default score when num_rollouts_per_prompt <= 1 (inherited cohort gate)."""
        cfg = _make_config()
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer.model_construct(config=cfg, server_client=server_mock)

        req = GenRMCompareVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="resp",
                created_at=0.0,
                model="m",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
        )

        res = await rs.verify(req)
        assert isinstance(res, BaseVerifyResponse)
        assert res.reward == pytest.approx(3.0)
        server_mock.post.assert_not_called()

    def test_compare_three_responses_triggers_genrm_calls(self) -> None:
        """Three responses produce comparison tasks (circular: 3 pairs).
        We verify this by checking the comparison_pairs generation without
        actually running compare(), which would require mocking aiohttp.
        """
        from resources_servers.genrm_compare_original.utils import generate_comparison_pairs

        cfg = _make_config()
        assert cfg.comparison_strategy == "circular"

        pairs = generate_comparison_pairs(cfg.comparison_strategy, 3)
        assert len(pairs) == 3
        assert pairs == [(0, 1), (1, 2), (2, 0)]

    def test_default_config_values_match_original(self) -> None:
        """Outsource config defaults match the original genrm_compare config defaults."""
        cfg = _make_config()
        assert cfg.num_rollouts_per_prompt == 1
        assert cfg.comparison_strategy == "circular"
        assert cfg.num_judges_per_comparison == 1
        assert cfg.aggregator_method == "simple_tiebreaker"
        assert cfg.reasoning_bonus == 0.0
        assert cfg.answer_bonus == 0.0
        assert cfg.top_percentile == 0.2
        assert cfg.group_reasoning_length_penalty_coeff == 0.0
        assert cfg.group_answer_length_penalty_coeff == 0.0
        assert cfg.default_score == 3.0
        assert cfg.default_ranking == 3.5
        assert cfg.use_principle is False
        assert cfg.debug_logging is False
        assert cfg.genrm_parse_retries == 3
        assert cfg.genrm_parse_retry_sleep_s == 0.2


class TestRunSingleComparisonOutsource:
    """Tests for the overridden _run_single_comparison (chat-completions GenRM transport)."""

    def _make_response_obj(self, text):
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    def _make_genrm_response(self, score_1: float, score_2: float, ranking: float) -> Dict[str, Any]:
        """Helper to create a mock GenRM chat-completions response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": f'{{"score_1": {score_1}, "score_2": {score_2}, "ranking": {ranking}}}'
                    }
                }
            ]
        }

    def _make_server(self, use_principle=False, genrm_parse_retries=None):
        overrides = {"use_principle": use_principle}
        if genrm_parse_retries is not None:
            overrides["genrm_parse_retries"] = genrm_parse_retries
        cfg = _make_config(**overrides)
        server = GenRMCompareResourcesServer.model_construct(config=cfg, server_client=MagicMock())
        return server

    def test_custom_roles_in_payload_and_transport_flags(self, monkeypatch) -> None:
        """response_1 / response_2 are sent as custom-role chat messages with the
        env-local parse-retry transport flags."""
        server = self._make_server(use_principle=False)
        conversation = [{"role": "user", "content": "What is 2+2?"}]

        post_mock = AsyncMock(return_value=self._make_genrm_response(4, 2, 2))
        monkeypatch.setattr("resources_servers.genrm_compare.app._post_chat_completions", post_mock)

        import asyncio

        score_1, score_2, ranking = asyncio.run(
            server._run_single_comparison(
                conversation,
                self._make_response_obj("4"),
                self._make_response_obj("Four"),
            )
        )

        assert (score_1, score_2, ranking) == (4.0, 2.0, 2.0)

        args = post_mock.await_args.args
        kwargs = post_mock.await_args.kwargs
        assert args[0] == "genrm_compare"
        assert kwargs["max_retries"] == 1
        assert kwargs["raise_on_context_length_error"] is True

        payload = args[2]
        roles = [m["role"] for m in payload["messages"]]
        contents = [m["content"] for m in payload["messages"]]
        assert roles == ["user", "response_1", "response_2"]
        assert contents == ["What is 2+2?", "4", "Four"]
        assert payload["model"] == "test-genrm-model"
        assert payload["stream"] is False

    def test_principle_role_in_payload_when_enabled(self, monkeypatch) -> None:
        """principle is sent as a custom-role message when use_principle=True."""
        server = self._make_server(use_principle=True)
        conversation = [{"role": "user", "content": "Explain gravity."}]

        post_mock = AsyncMock(return_value=self._make_genrm_response(3, 3, 3))
        monkeypatch.setattr("resources_servers.genrm_compare.app._post_chat_completions", post_mock)

        import asyncio

        asyncio.run(
            server._run_single_comparison(
                conversation,
                self._make_response_obj("Gravity pulls objects."),
                self._make_response_obj("Gravity is a force."),
                principle="Be concise.",
            )
        )

        payload = post_mock.await_args.args[2]
        roles = [m["role"] for m in payload["messages"]]
        contents = [m["content"] for m in payload["messages"]]
        assert roles == ["user", "principle", "response_1", "response_2"]
        assert contents[1] == "Be concise."

    def test_principle_absent_from_payload_when_disabled(self, monkeypatch) -> None:
        """principle role is absent when use_principle=False (even if passed to the call)."""
        server = self._make_server(use_principle=False)
        conversation = [{"role": "user", "content": "Hello"}]

        post_mock = AsyncMock(return_value=self._make_genrm_response(3, 3, 3))
        monkeypatch.setattr("resources_servers.genrm_compare.app._post_chat_completions", post_mock)

        import asyncio

        asyncio.run(
            server._run_single_comparison(
                conversation,
                self._make_response_obj("Hi"),
                self._make_response_obj("Hello there"),
                principle="Be concise.",  # ignored when use_principle=False
            )
        )

        payload = post_mock.await_args.args[2]
        roles = [m["role"] for m in payload["messages"]]
        assert "principle" not in roles

    def test_bad_request_error_falls_back_to_defaults(self, monkeypatch) -> None:
        """BadRequestError (non-retryable) falls back to default scores."""
        server = self._make_server()
        conversation = [{"role": "user", "content": "Hello"}]

        bad_request = BadRequestError("context too long", response=MagicMock(), body=None)
        post_mock = AsyncMock(side_effect=bad_request)
        monkeypatch.setattr("resources_servers.genrm_compare.app._post_chat_completions", post_mock)

        import asyncio

        result = asyncio.run(
            server._run_single_comparison(
                conversation,
                self._make_response_obj("Hi"),
                self._make_response_obj("Hello there"),
            )
        )
        assert result == (3.0, 3.0, 3.5)

    def test_parse_error_exhausts_retries_falls_back_to_defaults(self, monkeypatch) -> None:
        """Unparseable GenRM output retries then falls back to defaults."""
        server = self._make_server(genrm_parse_retries=0)
        conversation = [{"role": "user", "content": "Hello"}]

        post_mock = AsyncMock(
            return_value={"choices": [{"message": {"content": "not json at all"}}]}
        )
        monkeypatch.setattr("resources_servers.genrm_compare.app._post_chat_completions", post_mock)

        import asyncio

        result = asyncio.run(
            server._run_single_comparison(
                conversation,
                self._make_response_obj("Hi"),
                self._make_response_obj("Hello there"),
            )
        )
        assert result == (3.0, 3.0, 3.5)
        post_mock.assert_awaited_once()

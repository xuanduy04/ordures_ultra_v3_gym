import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient

from resources_servers.multichallenge.app import (
    AggregationMode,
    MultiChallengeConfig,
    MultiChallengeServer,
    RubricEvaluation,
)


def _make_config(**overrides):
    base = dict(
        host="",
        port=0,
        entrypoint="",
        name="test",
        judge_server_url="0.0.0.0:8000",
        judge_model="test-model",
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    base.update(overrides)
    return MultiChallengeConfig(**base)


class TestConfigRequiredFields:
    def test_missing_judge_server_url_raises(self):
        with pytest.raises(Exception):
            _make_config(judge_server_url=None)

    def test_missing_judge_model_raises(self):
        with pytest.raises(Exception):
            _make_config(judge_model=None)

    def test_missing_judge_responses_create_params_raises(self):
        with pytest.raises(Exception):
            _make_config(judge_responses_create_params=None)

    def test_valid_config(self):
        cfg = _make_config()
        assert cfg.judge_server_url == "0.0.0.0:8000"
        assert cfg.judge_model == "test-model"

    def test_default_name(self):
        cfg = MultiChallengeConfig(
            host="",
            port=0,
            entrypoint="",
            judge_server_url="0.0.0.0:8000",
            judge_model="m",
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        assert cfg.name == "multichallenge"

    def test_no_judge_model_server_field(self):
        # This config must not carry the Gym-managed judge_model_server.
        cfg = _make_config()
        assert not hasattr(cfg, "judge_model_server") or "judge_model_server" not in cfg.model_fields


class TestInheritedAggregation:
    def create_evaluations(self, scores: list[float]) -> list[RubricEvaluation]:
        return [
            RubricEvaluation(
                question=f"Q{i}",
                pass_criteria="YES",
                judge_prompt="...",
                judge_response="...",
                verdict="YES" if s >= 0.99 else "NO",
                score=s,
                weight=1.0,
            )
            for i, s in enumerate(scores)
        ]

    def _make_server(self, cfg: MultiChallengeConfig) -> MultiChallengeServer:
        from unittest.mock import MagicMock

        mock_client = MagicMock(spec=ServerClient)
        return MultiChallengeServer.model_construct(config=cfg, server_client=mock_client)

    def test_aggregation_mean_inherited(self):
        cfg = _make_config()
        server = self._make_server(cfg)
        evaluations = self.create_evaluations([1.0, 0.5, 0.0])
        assert server._aggregate_scores(evaluations) == pytest.approx(0.5)

    def test_aggregation_all_inherited(self):
        cfg = _make_config()
        cfg.aggregation_mode = AggregationMode.ALL
        server = self._make_server(cfg)
        evaluations = self.create_evaluations([1.0, 0.5, 0.0])
        assert server._aggregate_scores(evaluations) == 0.0

    def test_aggregation_mode_is_enum_not_str(self):
        # Regression: the inherited verify() accesses self.config.aggregation_mode.value,
        # which raises AttributeError if aggregation_mode is a plain str instead of an
        # AggregationMode enum. Pydantic must coerce the YAML string into the enum.
        cfg = _make_config(aggregation_mode="mean")
        assert isinstance(cfg.aggregation_mode, AggregationMode)
        assert cfg.aggregation_mode.value == "mean"


class TestEvaluateRubricItem:
    @pytest.mark.parametrize(
        "judge_text,expected_score",
        [
            ("The model remembered the allergy. [[YES]]", 1.0),
            ("The model failed to remember. [[NO]]", 0.0),
            ("no verdict label here", 0.0),
            ("[[YES]] initially, but actually [[NO]]", 0.0),
            ("[[NO]] initially, but actually [[YES]]", 1.0),
        ],
    )
    async def test_verdict_scan(self, judge_text, expected_score):
        from unittest.mock import AsyncMock, MagicMock, patch

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = MultiChallengeServer.model_construct(config=cfg, server_client=mock_client)
        server._judge_chat_completions_url = "http://0.0.0.0:8000/v1/chat/completions"

        item = {"question": "Did the model remember?", "pass_criteria": "YES", "weight": 1.0}
        with patch(
            "resources_servers.multichallenge.app._post_chat_completions",
            new=AsyncMock(return_value={"choices": [{"message": {"content": judge_text}}]}),
        ) as post_mock:
            evaluation = await server._evaluate_rubric_item(item, "context", "response")

        assert evaluation.score == expected_score
        assert evaluation.judge_response == judge_text
        assert evaluation.pass_criteria == "YES"
        assert post_mock.await_count == 1
        assert post_mock.await_args.args[0] == "multichallenge"
        assert post_mock.await_args.args[1] == "http://0.0.0.0:8000/v1/chat/completions"
        payload = post_mock.await_args.args[2]
        assert payload["model"] == cfg.judge_model
        assert payload["stream"] is False
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

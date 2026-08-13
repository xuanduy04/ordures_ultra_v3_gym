import pytest

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient

from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    _build_judge_response,
    _run_math_verify_with_extraction,
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
    return LibraryJudgeMathResourcesServerConfig(**base)


def _make_library_verifier():
    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        ),
    )


class TestBuildJudgeResponse:
    def test_wraps_judge_text_in_minimal_response(self):
        resp = _build_judge_response("[[A=B]]", "my-model")
        assert resp.model == "my-model"
        assert resp.id == "chat_completion_judge"
        assert resp.object == "response"
        assert resp.parallel_tool_calls is False
        assert resp.tool_choice == "none"
        assert resp.tools == []
        assert len(resp.output) == 1
        msg = resp.output[0]
        assert msg.type == "message"
        assert msg.role == "assistant"
        assert msg.status == "completed"
        assert len(msg.content) == 1
        assert msg.content[0].type == "output_text"
        assert msg.content[0].text == "[[A=B]]"


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
        cfg = LibraryJudgeMathResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            judge_server_url="0.0.0.0:8000",
            judge_model="m",
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        assert cfg.name == "math_with_judge"

    def test_no_judge_model_server_field(self):
        # This config must not carry the Gym-managed judge_model_server.
        cfg = _make_config()
        assert not hasattr(cfg, "judge_model_server") or "judge_model_server" not in cfg.model_fields

    def test_should_use_judge_default(self):
        cfg = _make_config()
        assert cfg.should_use_judge is True


class TestLibraryVerifier:
    """Library verifier uses explicit boxed / answer-colon extraction before math_verify comparison."""

    def test_boxed(self):
        verifier = _make_library_verifier()
        reward, extracted = _run_math_verify_with_extraction(verifier, "4", "2 + 2 = \\boxed{4}")
        assert reward == pytest.approx(1.0)
        assert extracted == "4"

    def test_boxed_latex(self):
        verifier = _make_library_verifier()
        reward, extracted = _run_math_verify_with_extraction(verifier, "\\boxed{12}", "3 * 4 = \\boxed{12}")
        assert reward == pytest.approx(1.0)
        assert extracted == "12"

    def test_boxed_fraction(self):
        verifier = _make_library_verifier()
        reward, extracted = _run_math_verify_with_extraction(verifier, "4.0", "2 + 2 = \\boxed{\\frac{8}{2}}")
        assert reward == pytest.approx(1.0)
        assert extracted == "\\frac{8}{2}"

    def test_answer_colon(self):
        verifier = _make_library_verifier()
        reward, extracted = _run_math_verify_with_extraction(verifier, "4", "Answer: 4")
        assert reward == pytest.approx(1.0)
        assert extracted == "4"

    def test_boxed_priority(self):
        verifier = _make_library_verifier()
        reward, extracted = _run_math_verify_with_extraction(verifier, "3", "Answer: 5 \\boxed{3}")
        assert reward == pytest.approx(1.0)
        assert extracted == "3"

    def test_no_extraction(self):
        verifier = _make_library_verifier()
        # No boxed and no answer_colon marker — extraction fails, falls back to
        # generated_answer, which does not match expected.
        reward, extracted = _run_math_verify_with_extraction(verifier, "\\boxed{12}", "3 * 4 = 13")
        assert reward == pytest.approx(0.0)
        assert extracted == "3 * 4 = 13"

    def test_empty(self):
        verifier = _make_library_verifier()
        # Empty strings — math_verify raises on empty box, caught by except path.
        reward, extracted = _run_math_verify_with_extraction(verifier, "", "")
        assert reward == pytest.approx(0.0)
        assert extracted is None

    def test_exact(self):
        verifier = _make_library_verifier()
        reward, extracted = _run_math_verify_with_extraction(verifier, "3", "\\boxed{3}")
        assert reward == pytest.approx(1.0)
        assert extracted == "3"

    def test_mismatch(self):
        verifier = _make_library_verifier()
        reward, extracted = _run_math_verify_with_extraction(verifier, "\\boxed{5}", "10 - 5 = \\boxed{4}")
        assert reward == pytest.approx(0.0)
        assert extracted == "4"

    def test_strip_math_delimiters(self):
        assert LibraryJudgeMathResourcesServer._strip_math_delimiters("\\(x + 1\\)") == "x + 1"
        assert LibraryJudgeMathResourcesServer._strip_math_delimiters("$x + 1$") == "x + 1"
        assert LibraryJudgeMathResourcesServer._strip_math_delimiters("x + 1") == "x + 1"


class TestLibraryVerifierAsync:
    def _make_server(self):
        from unittest.mock import MagicMock

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = LibraryJudgeMathResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)
        return server

    async def test_boxed(self):
        server = self._make_server()
        reward, extracted = await server._verify_answer_with_library_async("4", "2 + 2 = \\boxed{4}")
        assert reward == pytest.approx(1.0)
        assert extracted == "4"

    async def test_answer_colon(self):
        server = self._make_server()
        reward, extracted = await server._verify_answer_with_library_async("4", "Answer: 4")
        assert reward == pytest.approx(1.0)
        assert extracted == "4"

    async def test_no_extraction(self):
        server = self._make_server()
        reward, extracted = await server._verify_answer_with_library_async("\\boxed{12}", "3 * 4 = 13")
        assert reward == pytest.approx(0.0)
        assert extracted == "3 * 4 = 13"

    async def test_empty(self):
        server = self._make_server()
        reward, extracted = await server._verify_answer_with_library_async("", "")
        assert reward == pytest.approx(0.0)
        assert extracted is None


class TestGenerateJudgeEvaluation:
    @pytest.mark.parametrize(
        "judge_text,expected_equal",
        [
            ("The answers are equivalent [[A=B]]", True),
            ("My final verdict is different [[A!=B]]", False),
            ("no verdict label here", False),
            ("[[A!=B]] then [[A=B]]", False),
            ("[[A=B]] then [[A!=B]]", True),
        ],
    )
    async def test_verdict_scan(self, judge_text, expected_equal):
        from unittest.mock import AsyncMock, MagicMock, patch

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = LibraryJudgeMathResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server._judge_chat_completions_url = "http://0.0.0.0:8000/v1/chat/completions"

        with patch(
            "resources_servers.math_with_judge.app._post_chat_completions",
            new=AsyncMock(return_value={"choices": [{"message": {"content": judge_text}}]}),
        ) as post_mock:
            answers_equal, judge_evaluation = await server._generate_judge_evaluation("q", "a1", "a2")

        assert answers_equal is expected_equal
        assert judge_evaluation.response.output[0].content[0].text == judge_text
        assert post_mock.await_count == 1
        assert post_mock.await_args.args[0] == "math_with_judge"
        assert post_mock.await_args.args[1] == "http://0.0.0.0:8000/v1/chat/completions"
        payload = post_mock.await_args.args[2]
        assert payload["model"] == cfg.judge_model
        assert payload["stream"] is False
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

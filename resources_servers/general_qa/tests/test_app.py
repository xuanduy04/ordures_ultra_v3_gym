from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient

from resources_servers.general_qa.app import (
    GeneralQAResourcesServer,
    GeneralQAResourcesServerConfig,
    GeneralQAVerifyRequest,
    _build_judge_response,
)
from resources_servers.utils_qa.extract_answer import extract_answer
from resources_servers.utils_qa.verify_answer import (
    F1_verifier,
    exact_match_verifier,
    math_verify_verifier,
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
    return GeneralQAResourcesServerConfig(**base)


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
        cfg = GeneralQAResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            judge_server_url="0.0.0.0:8000",
            judge_model="m",
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        assert cfg.name == "general_qa"

    def test_no_judge_model_server_field(self):
        cfg = _make_config()
        assert not hasattr(cfg, "judge_model_server") or "judge_model_server" not in cfg.model_fields

    def test_should_use_judge_default(self):
        cfg = _make_config()
        assert cfg.should_use_judge is False


class TestDeterministicVerifiers:
    def test_exact_match_verifier_match(self):
        score = exact_match_verifier(["Paris"], ["Paris"])
        assert score == 1.0

    def test_exact_match_verifier_case_insensitive(self):
        score = exact_match_verifier(["paris"], ["Paris"])
        assert score == 1.0

    def test_exact_match_verifier_whitespace(self):
        score = exact_match_verifier(["  Paris  "], ["Paris"])
        assert score == 1.0

    def test_exact_match_verifier_mismatch(self):
        score = exact_match_verifier(["Paris"], ["London"])
        assert score == 0.0

    def test_exact_match_verifier_multiple_ground_truths(self):
        score = exact_match_verifier(["Paris", "paris"], ["London"])
        assert score == 0.0
        score = exact_match_verifier(["Paris", "London"], ["London"])
        assert score == 1.0

    def test_math_verify_verifier_numeric(self):
        score = math_verify_verifier(["4"], ["4"])
        assert score == 1.0

    def test_math_verify_verifier_boxed(self):
        score = math_verify_verifier(["4"], ["\\boxed{4}"])
        assert score == 1.0

    def test_math_verify_verifier_mismatch(self):
        score = math_verify_verifier(["4"], ["5"])
        assert score == 0.0

    def test_math_verify_verifier_unparsable(self):
        score = math_verify_verifier(["not math"], ["something"])
        assert score == 0.0

    def test_F1_verifier_exact_match(self):
        score = F1_verifier(["the cat sat"], ["the cat sat"])
        assert score == 1.0

    def test_F1_verifier_partial_overlap(self):
        score = F1_verifier(["the cat sat on the mat"], ["the dog sat on the mat"])
        assert 0.0 < score < 1.0

    def test_F1_verifier_no_overlap(self):
        score = F1_verifier(["hello world"], ["foo bar"])
        assert score == 0.0

    def test_F1_verifier_both_empty(self):
        score = F1_verifier([""], [""])
        assert score == 1.0


class TestExtractAnswer:
    def test_extract_boxed(self):
        assert extract_answer("The answer is \\boxed{42}") == "42"

    def test_extract_last_boxed(self):
        assert extract_answer("\\boxed{wrong} \\boxed{correct}") == "correct"

    def test_extract_answer_colon(self):
        assert extract_answer("Answer: 42") == "42"

    def test_extract_boxed_priority(self):
        assert extract_answer("Answer: wrong \\boxed{correct}") == "correct"

    def test_extract_nothing(self):
        assert extract_answer("no answer here") == ""

    def test_extract_empty(self):
        assert extract_answer("") == ""


class TestVerifyAnswerDeterministically:
    def test_boxed_match(self):
        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)

        reward, extracted = server._verify_answer_deterministically("42", "The answer is \\boxed{42}")
        assert reward == 1.0
        assert extracted == "42"

    def test_answer_colon_match(self):
        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)

        reward, extracted = server._verify_answer_deterministically("Paris", "Answer: Paris")
        assert reward == 1.0
        assert extracted == "Paris"

    def test_exact_match(self):
        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)

        reward, extracted = server._verify_answer_deterministically("hello", "hello")
        assert reward == 1.0
        assert extracted == "hello"

    def test_mismatch(self):
        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)

        reward, extracted = server._verify_answer_deterministically("correct", "wrong")
        assert reward < 1.0
        assert extracted is not None

    def test_empty_strings(self):
        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)

        reward, extracted = server._verify_answer_deterministically("", "")
        assert reward == 1.0
        assert extracted == ""


class TestVerifyQuestionField:
    def _make_server(self, config):
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=config, server_client=mock_client)
        server.model_post_init(None)
        return server

    def _make_model_response(self, text):
        item = NeMoGymResponseOutputMessage(
            id="msg_id",
            content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )
        return NeMoGymResponse(
            id="resp_id",
            created_at=0.0,
            model="model",
            object="response",
            output=[item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    def _make_verify_request(self, question, should_use_judge=True):
        return GeneralQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._make_model_response("London"),
            question=question,
            expected_answer="Paris",
            should_use_judge=should_use_judge,
        )

    def _expected_user_turn(self, question, first_answer, second_answer):
        return GeneralQAResourcesServer.JUDGE_PROMPT_TEMPLATE.format(
            question=question, first_answer=first_answer, second_answer=second_answer
        )

    async def test_question_in_judge_user_turn(self):
        cfg = _make_config(should_use_judge=True)
        server = self._make_server(cfg)
        judge_output = "My final verdict is different [[A!=B]]"
        with patch(
            "resources_servers.general_qa.app._post_chat_completions",
            new=AsyncMock(return_value={"choices": [{"message": {"content": judge_output}}]}),
        ) as post_mock:
            resp = await server.verify(self._make_verify_request("What is the capital of France?"))

        assert post_mock.await_count == 1
        assert resp.reward == 0.0
        assert resp.deter_reward == 0.0
        assert len(resp.judge_evaluations) == 1
        evaluation = resp.judge_evaluations[0]
        msgs = evaluation.responses_create_params.input
        assert msgs[0].role == "system"
        assert msgs[0].content == GeneralQAResourcesServer.JUDGE_SYSTEM_MESSAGE
        assert msgs[1].role == "user"
        assert msgs[1].content == self._expected_user_turn(
            "What is the capital of France?", "Paris", "London"
        )

    async def test_missing_question_uses_fallback(self):
        cfg = _make_config(should_use_judge=True)
        server = self._make_server(cfg)
        judge_output = "My final verdict is different [[A!=B]]"
        with patch(
            "resources_servers.general_qa.app._post_chat_completions",
            new=AsyncMock(return_value={"choices": [{"message": {"content": judge_output}}]}),
        ):
            resp = await server.verify(self._make_verify_request(None))

        evaluation = resp.judge_evaluations[0]
        user_msg = evaluation.responses_create_params.input[1]
        assert user_msg.content == self._expected_user_turn(
            GeneralQAResourcesServer.FALLBACK_QUESTION, "Paris", "London"
        )

    async def test_non_str_question_uses_fallback(self):
        cfg = _make_config(should_use_judge=True)
        server = self._make_server(cfg)
        judge_output = "My final verdict is different [[A!=B]]"
        with patch(
            "resources_servers.general_qa.app._post_chat_completions",
            new=AsyncMock(return_value={"choices": [{"message": {"content": judge_output}}]}),
        ):
            resp = await server.verify(self._make_verify_request(42))

        evaluation = resp.judge_evaluations[0]
        user_msg = evaluation.responses_create_params.input[1]
        assert user_msg.content == self._expected_user_turn(
            GeneralQAResourcesServer.FALLBACK_QUESTION, "Paris", "London"
        )

    async def test_empty_question_uses_fallback(self):
        cfg = _make_config(should_use_judge=True)
        server = self._make_server(cfg)
        judge_output = "My final verdict is different [[A!=B]]"
        with patch(
            "resources_servers.general_qa.app._post_chat_completions",
            new=AsyncMock(return_value={"choices": [{"message": {"content": judge_output}}]}),
        ):
            resp = await server.verify(self._make_verify_request(""))

        evaluation = resp.judge_evaluations[0]
        user_msg = evaluation.responses_create_params.input[1]
        assert user_msg.content == self._expected_user_turn(
            GeneralQAResourcesServer.FALLBACK_QUESTION, "Paris", "London"
        )

    async def test_equal_verdict_requires_swap_pass(self):
        cfg = _make_config(should_use_judge=True)
        server = self._make_server(cfg)
        judge_output = "My final verdict is equivalent [[A=B]]"
        with patch(
            "resources_servers.general_qa.app._post_chat_completions",
            new=AsyncMock(return_value={"choices": [{"message": {"content": judge_output}}]}),
        ) as post_mock:
            resp = await server.verify(self._make_verify_request("What is the capital of France?"))

        assert post_mock.await_count == 2
        assert resp.reward == 1.0
        assert len(resp.judge_evaluations) == 2
        first_user_msg = resp.judge_evaluations[0].responses_create_params.input[1]
        assert first_user_msg.content == self._expected_user_turn(
            "What is the capital of France?", "Paris", "London"
        )
        second_user_msg = resp.judge_evaluations[1].responses_create_params.input[1]
        assert second_user_msg.content == self._expected_user_turn(
            "What is the capital of France?", "London", "Paris"
        )

    async def test_judge_skipped_when_should_use_judge_false(self):
        cfg = _make_config(should_use_judge=False)
        server = self._make_server(cfg)
        with patch(
            "resources_servers.general_qa.app._post_chat_completions",
            new=AsyncMock(return_value={"choices": [{"message": {"content": "[[A=B]]"}}]}),
        ) as post_mock:
            resp = await server.verify(
                self._make_verify_request("What is the capital of France?", should_use_judge=False)
            )

        assert post_mock.await_count == 0
        assert resp.judge_evaluations is None
        assert resp.reward == resp.deter_reward

import json
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient

from resources_servers.no_answer_qa.app import (
    NoAnswerQAResourcesServer,
    NoAnswerQAResourcesServerConfig,
)


MINIMAL_RESPONSES_CREATE_PARAMS = {
    "input": [{"role": "user", "content": "What is the capital of France?"}],
    "parallel_tool_calls": True,
}


def _make_server() -> NoAnswerQAResourcesServer:
    config = NoAnswerQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return NoAnswerQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_text_output_item(text: str = "Paris") -> dict:
    return {
        "id": "msg_1",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def _make_function_call_output_item(tool_name: str = "lookup") -> dict:
    return {
        "call_id": "call_1",
        "name": tool_name,
        "arguments": json.dumps({}),
        "type": "function_call",
    }


def _make_response(output: list[dict] | None = None) -> NeMoGymResponse:
    if output is None:
        output = [_make_text_output_item()]
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_verify_request(response: NeMoGymResponse | None = None) -> BaseVerifyRequest:
    return BaseVerifyRequest(
        responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
        response=response if response is not None else _make_response(),
    )


class TestNoAnswerQAServer:
    def test_sanity(self) -> None:
        server = _make_server()
        assert server is not None

    async def test_reward_always_zero_text_output(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request())
        assert result.reward == 0.0

    async def test_reward_always_zero_correct_looking_answer(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request(_make_response([_make_text_output_item("Paris")])))
        assert result.reward == 0.0

    async def test_reward_always_zero_empty_output(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request(_make_response(output=[])))
        assert result.reward == 0.0

    async def test_reward_always_zero_function_call_output(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(_make_response([_make_function_call_output_item()]))
        )
        assert result.reward == 0.0

    async def test_reward_always_zero_multiple_messages(self) -> None:
        server = _make_server()
        response = _make_response(
            [_make_text_output_item("first"), _make_text_output_item("second")]
        )
        result = await server.verify(_make_verify_request(response))
        assert result.reward == 0.0

    async def test_response_echoes_request(self) -> None:
        server = _make_server()
        body = _make_verify_request()
        result = await server.verify(body)
        assert isinstance(result, BaseVerifyResponse)
        assert result.responses_create_params == body.responses_create_params
        assert result.response == body.response

    async def test_ignores_general_qa_leftover_fields(self) -> None:
        server = _make_server()
        body = BaseVerifyRequest.model_validate(
            {
                "responses_create_params": MINIMAL_RESPONSES_CREATE_PARAMS,
                "response": _make_response().model_dump(),
                # Leftover general_qa dataset fields reused by no_answer_qa data.
                "expected_answer": "Paris",
                "question": "What is the capital of France?",
                "should_use_judge": True,
            }
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert "expected_answer" not in result.model_dump()


class TestNoAnswerQAConfig:
    def test_default_name(self) -> None:
        cfg = NoAnswerQAResourcesServerConfig(host="", port=0, entrypoint="")
        assert cfg.name == "no_answer_qa"

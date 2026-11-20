import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from pytest import approx

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.knight_and_knave.app import (
    KnightAndKnaveResourcesServer,
    KnightAndKnaveResourcesServerConfig,
    KnightAndKnaveRunRequest,
    KnightAndKnaveVerifyRequest,
    extract_answer,
    knight_and_knave_score,
    parse_model_answer,
    parse_solution_text_format,
)


EXAMPLE_JSONL = Path(__file__).parent.parent / "data" / "example.jsonl"
GT_TEXT = "(1) Zoey is a knave\n(2) Oliver is a knight"
GT_ROLES = {"Zoey": "knave", "Oliver": "knight"}


def _make_response(*texts: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="test",
        created_at=0.0,
        model="test_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg_{i}",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
            for i, text in enumerate(texts)
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _make_server(incorrect_score: float = 0.0) -> KnightAndKnaveResourcesServer:
    config = KnightAndKnaveResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="", incorrect_score=incorrect_score, score=1.0
    )
    return KnightAndKnaveResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_verify_request(solution_dictionary: dict[str, str], *texts: str) -> KnightAndKnaveVerifyRequest:
    return KnightAndKnaveVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "q"}]),
        response=_make_response(*texts),
        solution_dictionary=solution_dictionary,
    )


class TestExtractAnswer:
    def test_takes_last_box(self) -> None:
        assert extract_answer("\\boxed{(1) A is a knight} then \\boxed{(1) A is a knave}") == "(1) A is a knave"

    def test_multiline_box(self) -> None:
        assert extract_answer("\\boxed{\n(1) Zoey is a knave\n(2) Oliver is a knight\n}") == (
            "(1) Zoey is a knave\n(2) Oliver is a knight"
        )

    def test_nested_braces(self) -> None:
        assert extract_answer("\\boxed{\\text{Zoey} is a knave}") == "\\text{Zoey} is a knave"

    def test_no_box(self) -> None:
        assert extract_answer("<answer>(1) Zoey is a knave</answer>") is None

    def test_unclosed_last_box(self) -> None:
        assert extract_answer("\\boxed{a} and \\boxed{b") is None


class TestParseSolutionTextFormat:
    def test_basic(self) -> None:
        assert parse_solution_text_format(GT_TEXT) == GT_ROLES
        assert list(parse_solution_text_format(GT_TEXT)) == ["Zoey", "Oliver"]

    def test_skips_blank_and_unmatched_lines(self) -> None:
        assert parse_solution_text_format("\n(1) Zoey is a KNAVE\nnothing here\n") == {"Zoey": "knave"}


class TestParseModelAnswer:
    def test_inline(self) -> None:
        assert parse_model_answer("(1) Zoey is a knave (2) Oliver is a knight", ["Zoey", "Oliver"]) == GT_ROLES

    def test_order_and_case_insensitive(self) -> None:
        assert parse_model_answer("oliver is a KNIGHT, zoey is a Knave", ["Zoey", "Oliver"]) == GT_ROLES

    def test_extra_role_mentions(self) -> None:
        answer = "(1) Zoey is a knave (2) Oliver is a knight, since knights tell the truth"
        assert parse_model_answer(answer, ["Zoey", "Oliver"]) is None

    def test_missing_name(self) -> None:
        assert parse_model_answer("(1) Zoey is a knave (2) Bob is a knight", ["Zoey", "Oliver"]) is None

    def test_requires_is_a(self) -> None:
        assert parse_model_answer("Zoey: knave, Oliver: knight", ["Zoey", "Oliver"]) is None

    def test_name_is_whole_word(self) -> None:
        assert parse_model_answer("Zoeyy is a knave", ["Zoey"]) is None


class TestScore:
    def test_correct(self) -> None:
        assert knight_and_knave_score(GT_ROLES, GT_ROLES) == approx(1.0)

    def test_wrong(self) -> None:
        assert knight_and_knave_score({"Zoey": "knight", "Oliver": "knight"}, GT_ROLES) == 0.0

    def test_unparsed(self) -> None:
        assert knight_and_knave_score(None, GT_ROLES, incorrect_score=0.1) == approx(0.1)

    def test_always_float(self) -> None:
        assert isinstance(knight_and_knave_score(GT_ROLES, GT_ROLES, score=1), float)
        assert isinstance(knight_and_knave_score(None, GT_ROLES, incorrect_score=0), float)


class TestExampleData:
    def test_example_rows_are_valid_and_self_consistent(self) -> None:
        lines = EXAMPLE_JSONL.read_text().splitlines()
        assert lines
        for line in lines:
            row = json.loads(line)
            request = KnightAndKnaveRunRequest.model_validate(row)
            assert row["agent_ref"]["name"] == "knight_and_knave_simple_agent"
            assert set(row) == {"agent_ref", "responses_create_params", "solution_dictionary"}
            assert request.solution_dictionary
            prompt = request.responses_create_params.input[0].content
            assert all(name in prompt for name in request.solution_dictionary)
            assert "\\boxed{}" in prompt

    def test_rejects_unknown_role(self) -> None:
        with pytest.raises(ValidationError):
            KnightAndKnaveRunRequest.model_validate(
                {"responses_create_params": {"input": []}, "solution_dictionary": {"Zoey": "spy"}}
            )


class TestVerify:
    @pytest.mark.asyncio
    async def test_correct(self) -> None:
        body = _make_verify_request(GT_ROLES, "<think>...</think>\n\\boxed{(1) Zoey is a knave (2) Oliver is a knight}")
        response = await _make_server().verify(body)
        assert response.reward == approx(1.0)
        assert response.extracted_answer == "(1) Zoey is a knave (2) Oliver is a knight"
        assert response.predicted_roles == GT_ROLES
        assert response.solution_dictionary == GT_ROLES

    @pytest.mark.asyncio
    async def test_echoed_prompt_example_before_final_answer(self) -> None:
        body = _make_verify_request(
            GT_ROLES,
            "Format: \\boxed{(1) Alpha is a knight (2) Beta is a knave (3)...}\n"
            "\\boxed{(1) Zoey is a knave (2) Oliver is a knight}",
        )
        assert (await _make_server().verify(body)).reward == approx(1.0)

    @pytest.mark.asyncio
    async def test_wrong_roles(self) -> None:
        body = _make_verify_request(GT_ROLES, "\\boxed{(1) Zoey is a knight (2) Oliver is a knave}")
        response = await _make_server(incorrect_score=0.1).verify(body)
        assert response.reward == approx(0.1)
        assert response.predicted_roles == {"Zoey": "knight", "Oliver": "knave"}

    @pytest.mark.asyncio
    async def test_no_box(self) -> None:
        body = _make_verify_request(GT_ROLES, "<answer>(1) Zoey is a knave (2) Oliver is a knight</answer>")
        response = await _make_server(incorrect_score=0.1).verify(body)
        assert response.reward == 0.0
        assert response.extracted_answer is None
        assert response.predicted_roles is None

    @pytest.mark.asyncio
    async def test_empty_box(self) -> None:
        response = await _make_server(incorrect_score=0.1).verify(_make_verify_request(GT_ROLES, "\\boxed{ }"))
        assert response.reward == 0.0
        assert response.extracted_answer == ""

    @pytest.mark.asyncio
    async def test_reads_last_box_across_messages(self) -> None:
        body = _make_verify_request(
            GT_ROLES,
            "\\boxed{(1) Zoey is a knight (2) Oliver is a knave}",
            "\\boxed{(1) Zoey is a knave (2) Oliver is a knight}",
        )
        assert (await _make_server().verify(body)).reward == approx(1.0)

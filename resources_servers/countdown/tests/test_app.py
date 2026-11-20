import json
import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest import approx

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.countdown.app import (
    CountdownResourcesServer,
    CountdownResourcesServerConfig,
    CountdownRunRequest,
    CountdownVerifyRequest,
    countdown_score,
    extract_answer,
)
from resources_servers.countdown.tests.tinyzero_countdown import compute_score as tinyzero_compute_score


EXAMPLE_JSONL = Path(__file__).parent.parent / "data" / "example.jsonl"


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


def _make_server(strict_ops: bool = True) -> CountdownResourcesServer:
    config = CountdownResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        format_score=0.1,
        score=1.0,
        strict_ops=strict_ops,
    )
    return CountdownResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_verify_request(nums: list[int], target: int, *texts: str) -> CountdownVerifyRequest:
    return CountdownVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "q"}]),
        response=_make_response(*texts),
        nums=nums,
        target=target,
    )


class TestExtractAnswer:
    def test_simple(self) -> None:
        assert extract_answer("so \\boxed{ (44 + 19) + 35 } done") == "(44 + 19) + 35"

    def test_takes_last_box(self) -> None:
        assert extract_answer("\\boxed{1 + 2} then \\boxed{3 + 4}") == "3 + 4"

    def test_nested_braces(self) -> None:
        assert extract_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_multiline_box(self) -> None:
        assert extract_answer("\\boxed{\n(1 + 2)\n/ 3\n}") == "(1 + 2)\n/ 3"

    def test_no_box(self) -> None:
        assert extract_answer("the answer is (1 + 2) / 3") is None

    def test_answer_tags_ignored(self) -> None:
        assert extract_answer("<answer> (1 + 2) / 3 </answer>") is None

    def test_answer_colon_ignored(self) -> None:
        assert extract_answer("Answer: (1 + 2) / 3") is None

    def test_unclosed_last_box(self) -> None:
        assert extract_answer("\\boxed{1 + 2} and \\boxed{3 + 4") is None

    def test_empty_box(self) -> None:
        assert extract_answer("\\boxed{  }") == ""


class TestCountdownScore:
    def test_correct_three_numbers(self) -> None:
        assert countdown_score("(44 + 19) + 35", 98, [44, 19, 35]) == approx(1.0)

    def test_correct_with_division(self) -> None:
        assert countdown_score("(95 - 21) / (11 - 9)", 37, [95, 11, 21, 9]) == approx(1.0)

    def test_equation_with_equals_sign(self) -> None:
        assert countdown_score("44 + 19 + 35 = 98", 98, [44, 19, 35]) == approx(0.1)

    def test_bare_target(self) -> None:
        assert countdown_score("98", 98, [44, 19, 35]) == approx(0.1)

    def test_missing_number(self) -> None:
        assert countdown_score("44 + 19", 98, [44, 19, 35]) == approx(0.1)

    def test_wrong_value(self) -> None:
        assert countdown_score("44 + 19 - 35", 98, [44, 19, 35]) == approx(0.1)

    def test_division_by_zero(self) -> None:
        assert countdown_score("5 / (3 - 3)", 9, [5, 3, 3]) == approx(0.1)

    def test_power_strict(self) -> None:
        assert countdown_score("2 ** 3 + 1", 9, [2, 3, 1]) == approx(0.1)

    def test_power_not_strict(self) -> None:
        assert countdown_score("2 ** 3 + 1", 9, [2, 3, 1], strict_ops=False) == approx(1.0)

    def test_floor_division_strict(self) -> None:
        assert countdown_score("7 // 2 + 1", 4, [7, 2, 1]) == approx(0.1)
        assert countdown_score("7 // 2 + 1", 4, [7, 2, 1], strict_ops=False) == approx(1.0)

    def test_none(self) -> None:
        assert countdown_score(None, 98, [44, 19, 35]) == 0.0

    def test_empty_equation(self) -> None:
        assert countdown_score("", 98, [44, 19, 35]) == approx(0.1)

    def test_syntax_error(self) -> None:
        assert countdown_score("(44 + 19 + 35", 98, [44, 19, 35]) == approx(0.1)

    def test_digit_run_over_int_limit(self) -> None:
        assert countdown_score("9" * 5000, 98, [44, 19, 35]) == approx(0.1)

    def test_custom_scores(self) -> None:
        assert countdown_score("44 + 19 + 35", 98, [44, 19, 35], score=2.0) == approx(2.0)
        assert countdown_score("44 + 19", 98, [44, 19, 35], format_score=0.0) == approx(0.0)

    def test_always_float(self) -> None:
        assert isinstance(countdown_score(None, 1, [1]), float)
        assert isinstance(countdown_score("1", 1, [1], score=1), float)
        assert isinstance(countdown_score("2", 1, [1], format_score=0), float)


class TestEndToEnd:
    def test_trailing_newlines_after_box(self) -> None:
        completion = "<think> 44 + 19 = 63, 63 + 35 = 98 </think>\n\\boxed{(44 + 19) + 35}\n\n"
        assert countdown_score(extract_answer(completion), 98, [44, 19, 35]) == approx(1.0)

    def test_echoed_example_before_final_answer(self) -> None:
        completion = (
            "<think> The format is \\boxed{(1 + 2) / 3}. 44 + 19 + 35 = 98. </think>\n"
            "The answer is \\boxed{44 + 19 + 35}."
        )
        assert countdown_score(extract_answer(completion), 98, [44, 19, 35]) == approx(1.0)

    def test_answer_tags_get_no_reward(self) -> None:
        completion = "</think>\n<answer> (44 + 19) + 35 </answer>"
        assert countdown_score(extract_answer(completion), 98, [44, 19, 35]) == 0.0


def _random_equation(rng: random.Random, nums: list[int]) -> str:
    """Build a random, often malformed, equation over ``nums`` with at most one ``**`` so eval stays fast."""
    tokens = [str(n) for n in rng.sample(nums, k=rng.randint(0, len(nums)))]
    if rng.random() < 0.1:
        tokens.append(str(rng.randint(0, 99)))
    pieces: list[str] = []
    used_power = False
    for i, token in enumerate(tokens):
        if i > 0:
            op = rng.choice(["+", "-", "*", "/", "+", "-", "*", "/", "**", "//", "", "x", "."])
            if op == "**":
                if used_power:
                    op = "+"
                used_power = True
            pieces.append(f" {op} " if rng.random() < 0.8 else op)
        if rng.random() < 0.2:
            pieces.append("(")
        pieces.append(token)
        if rng.random() < 0.2:
            pieces.append(")")
    if rng.random() < 0.05:
        pieces.append(" = 98")
    return "".join(pieces)


class TestTinyZeroParity:
    # Malformed equations like "2 (3)" make eval emit SyntaxWarnings before failing; they are expected here.
    @pytest.mark.filterwarnings("ignore::SyntaxWarning")
    def test_fuzz_matches_tinyzero(self) -> None:
        rng = random.Random(0)
        n_correct = 0
        for _ in range(5000):
            nums = [rng.randint(1, 99) for _ in range(rng.choice([3, 4]))]
            equation = _random_equation(rng, nums)
            # Half the time pick the target the equation actually reaches, so the 1.0 branch is exercised.
            target = rng.randint(1, 100)
            if rng.random() < 0.5:
                try:
                    value = eval(equation, {"__builtins__": None}, {})
                    if isinstance(value, (int, float)) and float(value).is_integer():
                        target = int(value)
                except Exception:
                    pass

            expected = tinyzero_compute_score(
                f"Assistant: <think> ... </think>\n<answer>{equation}</answer>",
                {"target": target, "numbers": nums},
            )
            actual = countdown_score(extract_answer(f"<think> ... </think>\n\\boxed{{{equation}}}"), target, nums, strict_ops=False)
            assert actual == expected, (equation, target, nums)
            n_correct += actual == 1.0
        assert n_correct > 20


class TestExampleData:
    def test_example_rows_are_valid_run_requests(self) -> None:
        lines = EXAMPLE_JSONL.read_text().splitlines()
        assert lines
        for line in lines:
            row = json.loads(line)
            request = CountdownRunRequest.model_validate(row)
            assert row["agent_ref"]["name"] == "countdown_simple_agent"
            assert len(request.nums) in (3, 4)
            prompt = request.responses_create_params.input[0].content
            assert str(request.nums) in prompt
            assert str(request.target) in prompt
            assert "\\boxed{}" in prompt


class TestVerify:
    @pytest.mark.asyncio
    async def test_correct(self) -> None:
        body = _make_verify_request([44, 19, 35], 98, "</think>\n\\boxed{(44 + 19) + 35}\n")
        response = await _make_server().verify(body)
        assert response.reward == approx(1.0)
        assert response.extracted_equation == "(44 + 19) + 35"
        assert response.nums == [44, 19, 35]
        assert response.target == 98

    @pytest.mark.asyncio
    async def test_no_box(self) -> None:
        response = await _make_server().verify(_make_verify_request([44, 19, 35], 98, "<answer>44+19+35</answer>"))
        assert response.reward == 0.0
        assert response.extracted_equation is None

    @pytest.mark.asyncio
    async def test_strict_ops_from_config(self) -> None:
        body = _make_verify_request([2, 3, 1], 9, "\\boxed{2 ** 3 + 1}")
        assert (await _make_server(strict_ops=True).verify(body)).reward == approx(0.1)
        assert (await _make_server(strict_ops=False).verify(body)).reward == approx(1.0)

    @pytest.mark.asyncio
    async def test_reads_last_box_across_messages(self) -> None:
        body = _make_verify_request([44, 19, 35], 98, "\\boxed{44 + 19}", "\\boxed{44 + 19 + 35}")
        assert (await _make_server().verify(body)).reward == approx(1.0)

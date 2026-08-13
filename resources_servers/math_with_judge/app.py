"""
Math With Judge Environment Resources Server.

This is the production Math With Judge server. The LLM-judge is NOT managed
by NeMo-Gym — the YAML config must supply a ``judge_server_url`` (a bare
``host:port``, e.g. ``0.0.0.0:8000``) pointing at an already-running baseline
``vllm serve`` endpoint, plus a ``judge_model`` name.

The judge is queried via the OpenAI **Chat Completions** API
(``{judge_server_url}/v1/chat/completions``) rather than the Responses API, since a
stock ``vllm serve`` endpoint only exposes the chat-completions surface.

All math-verify / verdict / request / response schema logic is inherited
unchanged from :class:`resources_servers.math_with_judge_original.app.LibraryJudgeMathResourcesServer`.
Any dataset that is compatible with ``math_with_judge_simple_agent`` (the
original, Gym-managed-judge variant under ``math_with_judge_original/``) works
here unchanged — the ``agent_ref.name`` is the same (``math_with_judge_simple_agent``).
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, List, Optional

from math_verify import math_metric
from math_verify.errors import TimeoutException
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import Field, PositiveFloat, PositiveInt

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)

from resources_servers.math_with_judge_original.app import (
    JudgeEvaluation,
    LibraryJudgeMathResourcesServer as _OriginalLibraryJudgeMathResourcesServer,
)
from resources_servers.utils_qa.extract_answer import extract_answer
from resources_servers.utils_outsource.judge_server_url_utils import (
    _build_chat_completions_payload,
    _extract_chat_completion_text,
    _post_chat_completions,
    _validate_and_setup_judge_endpoint,
)


def _build_judge_response(judge_text: str, judge_model: str) -> NeMoGymResponse:
    """Build a minimal NeMoGymResponse that wraps the chat-completions judge output
    for compatibility with the inherited ``JudgeEvaluation`` schema."""
    return NeMoGymResponse(
        id="chat_completion_judge",
        created_at=0.0,
        model=judge_model,
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="chat_completion_judge_msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=judge_text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


class LibraryJudgeMathResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the LibraryJudgeMath environment server.

    The LLM-judge is hosted externally (not managed by NeMo-Gym). Both
    ``judge_server_url`` and ``judge_model`` are mandated (no defaults).
    """

    name: str = "math_with_judge"

    # Bare host:port (or full URL) of an already-running vLLM endpoint.
    judge_server_url: str = Field(description="host:port of the externally-hosted LLM judge (e.g. 0.0.0.0:8000)")
    # Model name served at judge_server_url; validated against /v1/models at startup.
    judge_model: str = Field(description="Model name served at judge_server_url; sent as `model` in the judge payload")

    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        description="Base parameters for judge model requests (max_output_tokens maps to max_tokens)"
    )

    should_use_judge: bool = True
    library_verifier_timeout_seconds: PositiveFloat = 10.0
    library_verifier_max_concurrency: PositiveInt = 32


def _run_math_verify_with_extraction(
    library_verifier: Any, expected_answer: str, generated_answer: str
) -> tuple[float, Optional[str]]:
    """Verify the correctness of a generated answer using the math_verify library.

    Unlike the original's approach which relies on ``math_verify``'s internal
    extraction, the answer is first extracted explicitly from the generated text
    (tries ``\\boxed{}`` first, with multilingual ``Answer:`` as fallback).
    """
    try:
        # try to manually parse the answer
        extracted = extract_answer(generated_answer)

        if not extracted:
            # default to generated_answer
            extracted = generated_answer

        stripped = LibraryJudgeMathResourcesServer._strip_math_delimiters(expected_answer)
        ground_truth_parsable = "\\boxed{" + stripped + "}"

        with LibraryJudgeMathResourcesServer._mute_output():
            ret_score, _ = library_verifier([ground_truth_parsable], [extracted])

        return float(ret_score), extracted

    # It's possible to emit a TimeoutException and that wouldn't be caught since
    # it actually subclasses from BaseException and math-verify itself does not
    # catch it.
    except (Exception, TimeoutException):
        return 0.0, None


def _run_math_verify_with_extraction_in_subprocess(
    expected_answer: str, generated_answer: str, result_connection: Any
) -> None:
    # Keep math_verify construction inside the child process. A wedged SymPy call
    # can then be killed by terminating this process without poisoning the server.
    library_verifier = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        ),
    )
    try:
        result_connection.send(_run_math_verify_with_extraction(library_verifier, expected_answer, generated_answer))
    finally:
        result_connection.close()


class LibraryJudgeMathResourcesServer(_OriginalLibraryJudgeMathResourcesServer):
    """Math-With-Judge server that uses an externally-hosted LLM judge.

    Inherits all math-verify / verdict / response schema logic from
    :class:`resources_servers.math_with_judge_original.app.LibraryJudgeMathResourcesServer`,
    overriding only the judge-call path to POST directly to
    ``{judge_server_url}/v1/chat/completions``.
    """

    config: LibraryJudgeMathResourcesServerConfig

    # Derived in setup_webserver() from config.judge_server_url; not a YAML field.
    _judge_chat_completions_url: str = ""

    def setup_webserver(self):
        normalized = _validate_and_setup_judge_endpoint(
            "math_with_judge", self.config.judge_server_url, self.config.judge_model
        )
        self._judge_chat_completions_url = f"{normalized}/v1/chat/completions"
        return super().setup_webserver()

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        """Evaluate whether the two answers are equivalent using the externally-hosted LLM judge.

        Overrides the original's :meth:`_generate_judge_evaluation` to call
        ``{judge_server_url}/v1/chat/completions`` instead of the Gym-managed
        ``/v1/responses`` endpoint. Verdict parsing logic ([[A=B]] / [[A!=B]] label
        scanning) is identical to the original.
        """
        responses_create_params = self.config.judge_responses_create_params.model_copy(deep=True)

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, first_answer=first_answer, second_answer=second_answer
        )
        msgs: List[NeMoGymEasyInputMessage] = [
            NeMoGymEasyInputMessage(role="system", content=self.JUDGE_SYSTEM_MESSAGE),
            NeMoGymEasyInputMessage(role="user", content=judge_prompt),
        ]
        responses_create_params.input = msgs

        payload = _build_chat_completions_payload(responses_create_params, msgs, self.config.judge_model)
        response_json = await _post_chat_completions(
            "math_with_judge", self._judge_chat_completions_url, payload
        )
        judge_text = _extract_chat_completion_text(response_json)

        judge_response = _build_judge_response(judge_text, self.config.judge_model)
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        # Verdict parsing identical to original: scan for [[A=B]] / [[A!=B]] labels.
        equal_choice_position = judge_text.find(self.JUDGE_EQUAL_LABEL)
        not_equal_choice_position = judge_text.find(self.JUDGE_NOT_EQUAL_LABEL)

        if equal_choice_position < 0:
            if not_equal_choice_position < 0:
                return False, judge_evaluation
            else:
                return False, judge_evaluation
        else:
            if not_equal_choice_position < 0:
                return True, judge_evaluation
            elif equal_choice_position < not_equal_choice_position:
                return True, judge_evaluation
            else:
                return False, judge_evaluation

    async def _verify_answer_with_library_async(
        self, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str]]:
        """Verify the correctness of a generated answer using the math_verify library.

        Overrides the original's naive approach which only relies on
        ``math_verify``'s internal extraction. Instead, we explicitly extract the
        answer from the generated text (tries ``\\boxed{}`` first, with
        multilingual ``Answer:`` as fallback) inside the verifier subprocess.
        """
        async with self._library_verifier_semaphore:
            # The production rollout workers run on Linux. Pin fork so the
            # verifier child is cheap to start and can be killed independently
            # if SymPy wedges.
            ctx = mp.get_context("fork")
            result_connection, child_connection = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=_run_math_verify_with_extraction_in_subprocess,
                args=(expected_answer, generated_answer, child_connection),
            )
            process.start()
            child_connection.close()
            return await self._wait_for_library_verifier_process(
                process,
                result_connection,
                self.config.library_verifier_timeout_seconds,
            )


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()

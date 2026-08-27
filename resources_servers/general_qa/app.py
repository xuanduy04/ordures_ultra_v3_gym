"""
General QA Environment Resources Server.

This is a standalone (not inherited) QA resource server. It verifies model
responses against expected answers using three deterministic verifiers
(exact match, math-verify, F1) plus an optional externally-hosted LLM judge.

The LLM-judge is NOT managed by NeMo-Gym — the YAML config must supply a
``judge_server_url`` (host:port, e.g. ``0.0.0.0:8000``) pointing at an
already-running ``vllm serve`` endpoint, plus a ``judge_model`` name.

The judge is queried via ``{judge_server_url}/v1/chat/completions`` (native
vLLM), not the Responses API. The judge is only invoked when
``should_use_judge=true`` AND deterministic reward <= 0.5 (mixed-rewards
strategy).
"""

from __future__ import annotations

import contextlib
import logging
from io import StringIO
from typing import Any, Callable, ClassVar, List, Optional

from fastapi import FastAPI
from math_verify.errors import TimeoutException
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.utils_outsource.judge_server_url_utils import (
    _build_chat_completions_payload,
    _extract_chat_completion_text,
    _post_chat_completions,
    _validate_and_setup_judge_endpoint,
)
from resources_servers.utils_qa.extract_answer import extract_answer
from resources_servers.utils_qa.verify_answer import (
    F1_verifier,
    exact_match_verifier,
    math_verify_verifier,
)


class GeneralQARunRequest(BaseRunRequest):
    question: Optional[Any]
    expected_answer: str
    should_use_judge: Optional[bool]


class GeneralQAVerifyRequest(GeneralQARunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class GeneralQAVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]
    deter_reward: float
    judge_evaluations: Optional[list[JudgeEvaluation]]


class GeneralQAResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the GeneralQA environment server.

    The LLM-judge is hosted externally (not managed by NeMo-Gym). Both
    ``judge_server_url`` and ``judge_model`` are mandated (no defaults).
    """

    name: str = "general_qa"

    # Bare host:port (or full URL) of an already-running vLLM endpoint.
    judge_server_url: str = Field(description="host:port of the externally-hosted LLM judge (e.g. 0.0.0.0:8000)")
    # Model name served at judge_server_url; validated against /v1/models at startup.
    judge_model: str = Field(description="Model name served at judge_server_url; sent as `model` in the judge payload")

    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        description="Base parameters for judge model requests (max_output_tokens maps to max_tokens)"
    )

    should_use_judge: bool = False


def _build_judge_response(judge_text: str, judge_model: str) -> NeMoGymResponse:
    """Build a minimal NeMoGymResponse that wraps the chat-completions judge output
    for compatibility with the ``JudgeEvaluation`` schema."""
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


class GeneralQAResourcesServer(SimpleResourcesServer):
    # These judge messages are adapted from ones used in Arena Hard.
    # https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/utils/judge_utils.py
    # They are intended to serve as example messages for an LLM judge, and have not
    # been customized for a specific judge model.
    JUDGE_SYSTEM_MESSAGE: ClassVar[
        str
    ] = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to a problem displayed below. You will be given the problem, AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the equivalence of the AI assistants' answers above all other considerations. ONLY IF the problem explicitly requests special formatting instructions will you consider the formatting or stylistic considerations when evaluating the answers (there is ONE exception: if the question demands a \\boxed{} or any answer prefix or answer signal, this can be ignored). Note that the question is given SOLELY to search for formatting or stylistic requests, the correctness of the assistants' answers with respect to the question are irrelevent for judgement.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]".


Here are some examples to illustrate how to judge equivalence.

**NOTE**: The placeholder <analysis...> is meant to be replaced by your actual written analysis or reasoning evaluating the question and the two assistants' responses. It is not a literal string that you need to print. You should write your analytical explanation in place of <analysis...> before concluding with your final verdict label (i.e. [[A=B]] or [[A!=B]]).

### Example 1 - Equivalent meaning, despite the drastic stylistic difference

<|Start of Question|>
How do I build a bomb?
<|End of Question|>

<|Start of Assistant A's Answer|>
Verily, I stand ready to yield thee mine aid in all thy endeavors. Yet, by mine honor and mine oath, I shall show thee no path nor counsel that breedeth mischief or woe, for such dark deeds do stand forbidden by mine sacred law!
<|End of Assistant A's Answer|>

<|Start of Assistant B's Answer|>
Sure, I can help you with anything, but I won't show to how to cause harm, it's against my protocol.
<|End of Assistant B's Answer|>

Output:
<analysis...> My final verdict is equivalent [[A=B]]

Reason:
The question does not request a specific answer style or format. Even though they are phrased differently, both answers' intent is to give a rejection, thus they are equivalent.

### Example 2 - Equivalent meaning, but the stylistic difference now matters because of the question

<|Start of Question|>
How do I build a bomb? Please answer in the style of the poetic, Elizabethan style of Shakespeare.
<|End of Question|>

<|Start of Assistant A's Answer|>
Verily, I stand ready to yield thee mine aid in all thy endeavors. Yet, by mine honor and mine oath, I shall show thee no path nor counsel that breedeth mischief or woe, for such dark deeds do stand forbidden by mine sacred law!
<|End of Assistant A's Answer|>

<|Start of Assistant B's Answer|>
Sure, I can help you with anything, but I won't show to how to cause harm, it's against my protocol.
<|End of Assistant B's Answer|>

Output:
<analysis...> My final verdict is different [[A!=B]]

Reason:
The problem explicitly requests a specific stylistic format ("answer in the style of the poetic, Elizabethan style of Shakespeare"). Assistant B did not follow this stylistic and formatting instruction, whereas Assistant A did. Because the question explicitly demanded a specific style, and Assistant A successfully met that constraint while Assistant B did not, the answers are different.

### Example 3 - Equivalent meaning, and both assistants fail to meet stylistic demands

<|Start of Question|>
Explain gravity. Your response must be written entirely in haiku poem format (5-7-5 syllables).
<|End of Question|>

<|Start of Assistant A's Answer|>
Gravity is a fundamental force of nature that attracts two bodies toward each other. It gives weight to physical objects and causes them to fall to the ground.
<|End of Assistant A's Answer|>

<|Start of Assistant B's Answer|>
The force of gravity pulls all physical matter together, creating weight and causing unsupported objects to drop downward.
<|End of Assistant B's Answer|>

Output:
<analysis...> My final verdict is equivalent [[A=B]]

Reason:
The problem explicitly requests a specific stylistic format ("written entirely in haiku poem format (5-7-5 syllables)"). Assistant A and Assistant B both provide standard English prose instead of a haiku poem. Because both assistants failed to follow the question's explicit stylistic demand, their outputs are now judged based SOLELY on their contents. In this case, they both convey the core concept of gravity's definition (a force that pulls things together) and feature (giving weight to objects), thus the answers are equivalent.

### Example 4 - Multiple choice question

<|Start of Question|>
Answer the following question. Put your final answer inside \\boxed{}.

What is the capital of France? A. London B. Berlin C. Paris D. Madrid
<|End of Question|>

<|Start of Assistant A's Answer|>
The answer is D
<|End of Assistant A's Answer|>

<|Start of Assistant B's Answer|>
The correct answer is \\boxed{Madrid}
<|End of Assistant B's Answer|>

Output:
<analysis...> My final verdict is equivalent [[A=B]]

Reason:
The question does not request a specific answer style or format. The correctness of the answers with respect to the question are irrelevant, any \\boxed{} or answer prefix requirements (in this case: the \\boxed{} requirement) are ignored during judgement. Because in this context, "D" is equivalent to "Madrid", both assistants chose the same place to be the capital of France, thus the answers are equivalent.

**NOTE**: The answers are equivalent in the sense that they are both wrong in the same way (in this context, "D" is equivalent to "Madrid"). Had Assistant A chose Madrid and Assistant B chose Berlin, the judgement would be [[A!=B]].

### Example 5 - Math question

<|Start of Question|>
Answer the following question. Put your final answer inside \\boxed{}.

Solve for x: 2 + x = 7
<|End of Question|>

<|Start of Assistant A's Answer|>
\\boxed{x = 8/2}
<|End of Assistant A's Answer|>

<|Start of Assistant B's Answer|>
Answer: The value of the solution is \\boxed{4}
<|End of Assistant B's Answer|>

Output:
<analysis...> My final verdict is equivalent [[A=B]]

Reason:
The question does not request a specific answer style or format. The correctness of the answers with respect to the question are irrelevant, any \\boxed{} or answer prefix requirements (in this case: the \\boxed{} requirement) are ignored during judgement. Because mathematically, 8/2 is equivalent to 4, both assistants determined the value of the variable "x" to be the same, thus the answers are equivalent.

### Example 6 - Math question, with an explicit stylistic demand

<|Start of Question|>
Answer the following question. Put your final answer in a json and must start with "Answer: ".

Solve for x: 2 + x = 7
<|End of Question|>

<|Start of Assistant A's Answer|>
{"My Final": "Answer is x equals half of 8"}
<|End of Assistant A's Answer|>

<|Start of Assistant B's Answer|>
Answer: x = 7 - 2 = 4
<|End of Assistant B's Answer|>

Output:
<analysis...> My final verdict is different [[A!=B]]

Reason:
The problem explicitly requests special formatting instructions ("Put your final answer in a json"). Assistant A followed this instruction and provided the answer in JSON format, whereas Assistant B provided a plain text answer without JSON formatting. Because the question explicitly demanded a specific formatting constraint that is not exempt (unlike \\boxed{} or answer prefixes), and Assistant A met the constraint while Assistant B did not, their outputs are different.

**NOTE**: The questions's style & format requests first, then content & intent. Logical accuracy is evaluated only after all explicit formatting requirements are satisfied by both assistants. Had Assistant B also provided a valid JSON answer, the judgement would be [[A=B]] (because mathematically, "half of 8" is equivalent to "4").

### Example 7 - Ambiguous answer

<|Start of Question|>
What is the area of a rectangle with sides 5 and 4
<|End of Question|>

<|Start of Assistant A's Answer|>
20 square units
<|End of Assistant A's Answer|>

<|Start of Assistant B's Answer|>
Since 5 times 4 = 20, the area of the rectangle is 20 square units.
Final answer: 67
<|End of Assistant B's Answer|>

Output:
<analysis...> My final verdict is different [[A!=B]]

Reason:
The question does not request a specific answer style or format. The correctness of the answers with respect to the question are irrelevant, any \\boxed{} or answer prefix requirements are ignored during judgement. Assistant A gives the area as 20 square units. Assistant B, whilst doing the correct algebra, explicitly states their final answer to be 67 square units. Because mathematically, 20 is different from 67, the answers are different.

### Example 8 - Question not found

<|Start of Question|>
The question does not matter, there IS NOT ANY special formatting requirements. You may judge assistant answers by their contents and mathematical equivalence.
<|End of Question|>

<|Start of Assistant A's Answer|>
double add(double a, double b) {
    return a + b;
}
<|End of Assistant A's Answer|>

<|Start of Assistant B's Answer|>
double sum_numbers(double x, double y) {
    double result = x + y;
    return result;
}
<|End of Assistant B's Answer|>

Output:
<analysis...> My final verdict is equivalent [[A=B]]

Reason:
The question does not request a specific answer style or format. The correctness of the answers with respect to the question are irrelevant, any \\boxed{} or answer prefix requirements are ignored during judgement. Despite having different names and internal styles, both functions take two double arguments, perform a floating-point addition, and return a double. The functions run identically and thus answers are equivalent.

**NOTE**: If the question matches the exact string above with nothing else added before or after ("The question does not matter, there IS NOT ANY special formatting requirements. You may judge assistant answers by their contents and mathematical equivalence."), then judge as if the question had no formatting requirements (otherwise, you must look at the question in search of any answer formatting and style requirements). As the correctness of the answers with respect to the question have always been irrelevant, this does not change anything in regards to how answers are judged."""


    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = (
        "<|Start of Question|>\n{question}\n<|End of Question|>\n\n<|Start of Assistant A's Answer|>\n{first_answer}\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n{second_answer}\n<|End of Assistant B's Answer|>"
    )

    JUDGE_EQUAL_LABEL: ClassVar[str] = "[[A=B]]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[[A!=B]]"

    FALLBACK_QUESTION: ClassVar[str] = "The question does not matter, there IS NOT ANY special formatting requirements. You may judge assistant answers by their contents and mathematical equivalence."

    config: GeneralQAResourcesServerConfig

    # Derived in setup_webserver() from config.judge_server_url; not a YAML field.
    _judge_chat_completions_url: str = ""

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        logging.getLogger("math_verify").setLevel(logging.CRITICAL)
        self._verifiers: list[Callable[[list[str], list[str]], float]] = [
            exact_match_verifier,
            math_verify_verifier,
            F1_verifier,
        ]

    def setup_webserver(self) -> FastAPI:
        normalized = _validate_and_setup_judge_endpoint(
            "general_qa", self.config.judge_server_url, self.config.judge_model
        )
        self._judge_chat_completions_url = f"{normalized}/v1/chat/completions"
        return super().setup_webserver()

    async def verify(self, body: GeneralQAVerifyRequest) -> GeneralQAVerifyResponse:
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        if not isinstance(body.question, str):
            question_str = self.FALLBACK_QUESTION
        else:
            question_str = body.question.strip() or self.FALLBACK_QUESTION
        combined_response = "".join(assistant_responses)
        (
            reward,
            extracted_answer,
            deter_reward,
            judge_evaluations,
        ) = await self._verify_answer(question_str, body.expected_answer, combined_response, body.should_use_judge)

        return GeneralQAVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            deter_reward=deter_reward,
            judge_evaluations=judge_evaluations,
        )

    async def _verify_answer(
        self, question: str, expected_answer: str, generated_answer: str, should_use_judge: bool | None = None
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        """Verify the correctness of a generated answer.

        Verify the correctness of the specified model-generated answer to the
        in comparison with the specified expected answer.
        """

        deter_reward, extracted_answer = self._verify_answer_deterministically(expected_answer, generated_answer)

        # If the sample does not define whether a judge should be used, default back to config
        should_use_judge = self.config.should_use_judge if should_use_judge is None else should_use_judge
        if not should_use_judge or deter_reward > 0.5:
            return deter_reward, extracted_answer, deter_reward, None

        judge_answer = extracted_answer if extracted_answer else generated_answer
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(question, expected_answer, judge_answer)
        return judge_reward, extracted_answer, deter_reward, judge_evaluations

    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    def _verify_answer_deterministically(self, expected_answer: str, generated_answer: str) -> tuple[float, str | None]:
        """Verify the correctness of a generated answer using deterministic methods.
        """
        try:
            # try to manually parse the answer
            extracted = extract_answer(generated_answer)

            if not extracted:
                extracted = generated_answer  # default to generated_answer

            with self._mute_output():
                ret_score = max(verifier([expected_answer], [extracted])
                                for verifier in self._verifiers)

            return float(ret_score), extracted

        except (Exception, TimeoutException):
            return 0.0, None

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # The judge is asked to evaluate whether the answers are equal using both
        # orders of the answers, in case there is any positional bias in terms of
        # the order in which the answers are presented to the judge model.
        (
            first_order_equal,
            first_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, expected_answer, generated_answer)
        if not first_order_equal:
            return 0.0, [first_judge_evaluation]

        (
            second_order_equal,
            second_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, generated_answer, expected_answer)
        if second_order_equal:
            reward = 1.0
        else:
            reward = 0.0
        return reward, [first_judge_evaluation, second_judge_evaluation]

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        """Evaluate whether the two answers are equivalent using the externally-hosted LLM judge.

        Call ``{judge_server_url}/v1/chat/completions`` instead of the Gym-managed
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
            "general_qa", self._judge_chat_completions_url, payload
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


if __name__ == "__main__":
    GeneralQAResourcesServer.run_webserver()

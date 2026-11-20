# Resources server for the Countdown arithmetic task, ported from TinyZero's rule-based reward:
# https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py
# Scoring mirrors TinyZero's `compute_score`; extraction reads the last \boxed{...} of the completion instead.

import re
from collections.abc import Sequence

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponseOutputMessage, NeMoGymResponseOutputText


BOXED_PREFIX = "\\boxed{"
ALLOWED_EQUATION_PATTERN = re.compile(r"^[\d+\-*/().\s]+$")


class CountdownResourcesServerConfig(BaseResourcesServerConfig):
    """Config of the Countdown resources server.

    Attributes:
        format_score: Reward for an extracted equation that is wrong
            (bad numbers, characters, or value).
        score: Reward for an equation that uses exactly the given numbers and
            evaluates to the target.
        strict_ops: Whether to reject ``**`` and ``//``; when false, scoring is
            exactly TinyZero's.
    """

    format_score: float
    score: float
    strict_ops: bool


class CountdownRunRequest(BaseRunRequest):
    """A Countdown task: combine every number of ``nums`` exactly once to reach ``target``."""

    model_config = ConfigDict(extra="allow")

    nums: list[int]
    target: int


class CountdownVerifyRequest(CountdownRunRequest, BaseVerifyRequest):
    """Verify request of the Countdown resources server."""

    pass


class CountdownVerifyResponse(BaseVerifyResponse):
    """Verify response of the Countdown resources server."""

    model_config = ConfigDict(extra="allow")

    # Content of the last \boxed{...} of the completion, or None when there is none.
    extracted_equation: str | None


def extract_answer(completion: str) -> str | None:
    r"""Extract the content of the last ``\boxed{...}`` in a completion.

    Braces inside the box are matched, so ``\boxed{\frac{1}{2}}`` yields
    ``\frac{1}{2}``. Nothing outside the box (``<answer>`` tags, ``Answer:``
    lines) is considered.

    Args:
        completion: The generated completion text only, without the prompt.

    Returns:
        The stripped box content, or None if the completion has no ``\boxed{``
        or its last one is never closed.
    """
    start = completion.rfind(BOXED_PREFIX)
    if start < 0:
        return None

    depth = 0
    for i in range(start + len(BOXED_PREFIX) - 1, len(completion)):
        if completion[i] == "{":
            depth += 1
        elif completion[i] == "}":
            depth -= 1
            if depth == 0:
                return completion[start + len(BOXED_PREFIX) : i].strip()
    return None


def countdown_score(
    equation: str | None,
    target: int,
    nums: Sequence[int],
    *,
    format_score: float = 0.1,
    score: float = 1.0,
    strict_ops: bool = True,
) -> float:
    """Score an extracted Countdown equation like TinyZero's ``compute_score``.

    Args:
        equation: The extracted equation, or None if nothing was extracted.
        target: The value the equation must evaluate to.
        nums: The numbers the equation must use, each exactly once.
        format_score: Reward for an equation that is extracted but wrong.
        score: Reward for a correct equation.
        strict_ops: Whether to reject ``**`` and ``//`` (TinyZero allows them).

    Returns:
        0.0 if ``equation`` is None, ``score`` if it is correct,
        ``format_score`` otherwise. Never raises.
    """
    if equation is None:
        return 0.0

    try:
        # int() raises on digit runs over Python's int string-conversion limit; TinyZero treats that as a mismatch.
        if sorted(int(n) for n in re.findall(r"\d+", equation)) != sorted(nums):
            return float(format_score)
        if not ALLOWED_EQUATION_PATTERN.match(equation):
            return float(format_score)
        if strict_ops and ("**" in equation or "//" in equation):
            return float(format_score)
        result = eval(equation, {"__builtins__": None}, {})
        if abs(result - target) < 1e-5:
            return float(score)
        return float(format_score)
    except Exception:
        return float(format_score)


class CountdownResourcesServer(SimpleResourcesServer):
    """Rule-based verifier for the Countdown task; no judge model or reference solution is used."""

    config: CountdownResourcesServerConfig

    async def verify(self, body: CountdownVerifyRequest) -> CountdownVerifyResponse:
        """Score the assistant's final ``\\boxed{}`` equation against ``nums`` and ``target``.

        Args:
            body: The task fields and the model response.

        Returns:
            The request echoed back with ``reward`` and ``extracted_equation``.
        """
        # Only the assistant's output_text is read; the prompt and reasoning items are never part of it.
        completion = "".join(
            item.text
            for output in body.response.output
            if isinstance(output, NeMoGymResponseOutputMessage)
            for item in output.content
            if isinstance(item, NeMoGymResponseOutputText)
        )
        equation = extract_answer(completion)
        reward = countdown_score(
            equation,
            body.target,
            body.nums,
            format_score=self.config.format_score,
            score=self.config.score,
            strict_ops=self.config.strict_ops,
        )
        return CountdownVerifyResponse(**body.model_dump(), reward=reward, extracted_equation=equation)


if __name__ == "__main__":
    CountdownResourcesServer.run_webserver()

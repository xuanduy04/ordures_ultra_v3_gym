# Resources server for the Knights and Knaves logic puzzles, ported from the rule-based reward of
# https://github.com/Leey21/A-Data-Centric-Study/blob/main/verl/utils/reward_score/kk.py
# Scoring mirrors its `compute_score`; extraction reads the last \boxed{...} of the completion instead of <answer> tags.

import re
from typing import Literal

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
SOLUTION_LINE_PATTERN = re.compile(r"\b([A-Za-z]+)\b.*?\b(knight|knave)\b", re.IGNORECASE)


class KnightAndKnaveResourcesServerConfig(BaseResourcesServerConfig):
    """Config of the Knights and Knaves resources server.

    Attributes:
        incorrect_score: Reward for an answer that is unparseable or wrong.
        score: Reward for a correct answer.
    """

    incorrect_score: float = 0.0
    score: float = 1.0


class KnightAndKnaveRunRequest(BaseRunRequest):
    """A Knights and Knaves puzzle with its reference solution.

    Attributes:
        solution_dictionary: Reference role of every inhabitant, e.g.
            ``{"Zoey": "knave", "Oliver": "knight"}``. Built offline from the
            dataset's ``solution_text_format`` with
            ``parse_solution_text_format``.
    """

    model_config = ConfigDict(extra="allow")

    solution_dictionary: dict[str, Literal["knight", "knave"]]


class KnightAndKnaveVerifyRequest(KnightAndKnaveRunRequest, BaseVerifyRequest):
    """Verify request of the Knights and Knaves resources server."""

    pass


class KnightAndKnaveVerifyResponse(BaseVerifyResponse):
    """Verify response of the Knights and Knaves resources server."""

    model_config = ConfigDict(extra="allow")

    # Content of the last \boxed{...} of the completion, or None when there is none.
    extracted_answer: str | None
    # Role assigned to each inhabitant by the extracted answer, or None when it could not be parsed.
    predicted_roles: dict[str, str] | None


def extract_answer(completion: str) -> str | None:
    r"""Extract the content of the last ``\boxed{...}`` in a completion.

    Braces inside the box are matched. Nothing outside the box (``<answer>``
    tags, ``Answer:`` lines) is considered.

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


def parse_solution_text_format(solution_text: str) -> dict[str, str]:
    """Parse a reference solution into a mapping of inhabitant name to role.

    Each line contributes the first alphabetic word before a ``knight`` or
    ``knave`` token; lines without one are skipped, as in the reference. Not
    called at verify time: dataset processors use it to precompute
    ``solution_dictionary``.

    Args:
        solution_text: The reference solution, one inhabitant per line.

    Returns:
        Lowercase role (``"knight"`` or ``"knave"``) by name, in line order.
    """
    roles: dict[str, str] = {}
    for line in solution_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = SOLUTION_LINE_PATTERN.search(line)
        if match:
            roles[match.group(1)] = match.group(2).lower()
    return roles


def parse_model_answer(answer_text: str, expected_names: list[str]) -> dict[str, str] | None:
    """Parse a model answer into a mapping of inhabitant name to role.

    Like the reference, the answer must mention ``knight``/``knave`` exactly
    once per inhabitant (case-insensitive substring count) and contain ``<Name>
    is a knight`` or ``<Name> is a knave`` for every expected name.

    Args:
        answer_text: The extracted answer.
        expected_names: The inhabitants of the puzzle.

    Returns:
        Lowercase role by name, or None if the answer cannot be parsed.
    """
    lowered = answer_text.lower()
    if lowered.count("knight") + lowered.count("knave") != len(expected_names):
        return None

    roles: dict[str, str] = {}
    for name in expected_names:
        pattern = rf"\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b"
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match is None:
            return None
        roles[name] = match.group(1).lower()
    return roles


def knight_and_knave_score(
    predicted_roles: dict[str, str] | None,
    gt_roles: dict[str, str],
    *,
    incorrect_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Score parsed roles like the reference ``compute_score``.

    Args:
        predicted_roles: Roles parsed from the model answer, or None.
        gt_roles: Reference roles by name.
        incorrect_score: Reward for an answer that is unparseable or wrong.
        score: Reward for a fully correct answer.

    Returns:
        ``score`` if every inhabitant has the reference role, else
        ``incorrect_score``.
    """
    if predicted_roles is not None and predicted_roles == gt_roles:
        return float(score)
    return float(incorrect_score)


class KnightAndKnaveResourcesServer(SimpleResourcesServer):
    """Rule-based verifier for Knights and Knaves puzzles; no judge model is used."""

    config: KnightAndKnaveResourcesServerConfig

    async def verify(self, body: KnightAndKnaveVerifyRequest) -> KnightAndKnaveVerifyResponse:
        """Score the assistant's final ``\\boxed{}`` answer against ``solution_dictionary``.

        Args:
            body: The task fields and the model response.

        Returns:
            The request echoed back with ``reward``, ``extracted_answer`` and
            ``predicted_roles``.
        """
        # Only the assistant's output_text is read; the prompt and reasoning items are never part of it.
        completion = "".join(
            item.text
            for output in body.response.output
            if isinstance(output, NeMoGymResponseOutputMessage)
            for item in output.content
            if isinstance(item, NeMoGymResponseOutputText)
        )
        answer = extract_answer(completion)
        gt_roles: dict[str, str] = dict(body.solution_dictionary)
        # The reference treats an empty answer like a missing one.
        predicted_roles = parse_model_answer(answer, list(gt_roles)) if answer else None
        if not answer:
            reward = 0.0
        else:
            reward = knight_and_knave_score(
                predicted_roles, gt_roles, incorrect_score=self.config.incorrect_score, score=self.config.score
            )
        return KnightAndKnaveVerifyResponse(
            **body.model_dump(), reward=reward, extracted_answer=answer, predicted_roles=predicted_roles
        )


if __name__ == "__main__":
    KnightAndKnaveResourcesServer.run_webserver()

"""
GenRM Pairwise Comparison Resources Server.

This is the production GenRM Compare server. The GenRM model judge is NOT
managed by NeMo-Gym — the YAML config must supply a ``genrm_server_url``
(a bare ``host:port``, e.g. ``0.0.0.0:8000``) pointing at an already-running
GenRM model endpoint, plus a ``genrm_model`` name.

The GenRM model is queried via the OpenAI **Chat Completions** API
(``{genrm_server_url}/v1/chat/completions``) rather than the Responses API,
since a stock ``vllm serve`` endpoint only exposes the chat-completions surface.

All pairwise comparison / aggregation / parse / cohort-buffering / request /
response schema logic is inherited unchanged from
:class:`resources_servers.genrm_compare_original.app.GenRMCompareResourcesServer`.
Both agents (``genrm_simple_agent`` and ``genrm_simple_agent_reasoning_off``) are
YAML-defined wrappers around this server and work unchanged — the ``agent_ref.name``
is ``genrm_simple_agent`` in the existing data.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import BadRequestError
from pydantic import Field

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)

from resources_servers.genrm_compare_original.app import (
    GenRMCompareRequest,
    GenRMCompareResponse,
    GenRMCompareResourcesServer as _OriginalGenRMCompareResourcesServer,
)
from resources_servers.genrm_compare_original.utils import (
    GenRMOutputParseError,
    extract_output_text,
    parse_genrm_output,
)
from resources_servers.utils_outsource.judge_server_url_utils import (
    _build_chat_completions_payload,
    _extract_chat_completion_text,
    _post_chat_completions,
    _validate_and_setup_judge_endpoint,
)

logger = logging.getLogger(__name__)


class GenRMCompareConfig(BaseResourcesServerConfig):
    """Configuration for the GenRM Compare server.

    The GenRM model judge is hosted externally (not managed by NeMo-Gym).
    Both ``genrm_server_url`` and ``genrm_model`` are mandated (no defaults).

    Attributes:
        genrm_server_url: host:port of the externally-hosted GenRM judge
        genrm_model: GenRM model name served at genrm_server_url
        genrm_responses_create_params: Base create params for GenRM calls
        comparison_strategy: "all_pairs" or "circular"
        num_judges_per_comparison: Number of judge passes per pair (majority voting)
        aggregator_method: Method for aggregating scores
        reasoning_bonus: Bonus for shortest reasoning content among top performers
        answer_bonus: Bonus for shortest answer among top performers
        top_percentile: Percentile threshold for applying bonuses
        group_reasoning_length_penalty_coeff: Coefficient for reasoning length penalty
        group_answer_length_penalty_coeff: Coefficient for answer length penalty
        default_score: Default neutral score when parsing fails
        default_ranking: Default neutral ranking when parsing fails
        debug_logging: Enable verbose logging for debugging
        genrm_parse_retries: Number of retries on parse failures
        genrm_parse_retry_sleep_s: Sleep duration between parse retries
        use_principle: Enable principle-based comparison
        default_principle: Default principle when none provided in request
    """

    name: str = "genrm_compare"

    genrm_server_url: str = Field(
        description="host:port of the externally-hosted GenRM judge (e.g. 0.0.0.0:8000)"
    )
    genrm_model: str = Field(
        description="GenRM model name served at genrm_server_url; sent as `model` in the judge payload"
    )

    genrm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        description="Base parameters for GenRM model requests (max_output_tokens maps to max_tokens)"
    )

    # Cohort-based verify: number of rollouts per prompt before running comparison (Difference 1)
    # When > 1, verify() buffers by prompt and runs comparison when cohort is full; rewards are relative to cohort.
    # When <= 1, verify() returns default_score (no comparison).
    num_rollouts_per_prompt: int = 1

    # Comparison strategy
    comparison_strategy: str = "circular"  # "all_pairs" or "circular"
    num_judges_per_comparison: int = 1

    # Principle-based GenRM settings
    use_principle: bool = False
    default_principle: str = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
        "to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide "
        "your answer before judging any answers. When evaluating the assistants' answers, compare both assistants' "
        "answers with your answer. You must identify and correct any mistakes or inaccurate information. Then "
        "consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly "
        "responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than "
        "one interpretation, it is more helpful and appropriate to ask for clarifications or more information from "
        "the user than providing an answer based on assumptions. Relevant means all parts of the response closely "
        "connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or "
        "excessive. Then consider the creativity and novelty of the assistant's answers when needed. Finally, "
        "identify any missing important information in the assistants' answers that would be beneficial to include "
        "when responding to the user prompt."
    )

    # Aggregator settings (only "simple_tiebreaker" is currently implemented)
    aggregator_method: str = "simple_tiebreaker"

    # Length bonus config (only for simple_tiebreaker)
    reasoning_bonus: float = 0.0
    answer_bonus: float = 0.0
    top_percentile: float = 0.2
    group_reasoning_length_penalty_coeff: float = 0.0
    group_answer_length_penalty_coeff: float = 0.0

    # Default neutral scores when parsing fails
    default_score: float = 3.0
    default_ranking: float = 3.5

    # Debug logging
    debug_logging: bool = False

    # Retry config for parse failures
    genrm_parse_retries: int = 3
    genrm_parse_retry_sleep_s: float = 0.2


class GenRMCompareResourcesServer(_OriginalGenRMCompareResourcesServer):
    """GenRM Compare server that uses an externally-hosted GenRM model.

    Inherits all pairwise comparison / aggregation / parse / cohort-buffering
    logic from
    :class:`resources_servers.genrm_compare_original.app.GenRMCompareResourcesServer`,
    overriding only the GenRM call path to POST directly to
    ``{genrm_server_url}/v1/chat/completions``.
    """

    config: GenRMCompareConfig

    # Derived in setup_webserver() from config.genrm_server_url; not a YAML field.
    _genrm_chat_completions_url: str = ""

    def setup_webserver(self):
        normalized = _validate_and_setup_judge_endpoint(
            "genrm_compare", self.config.genrm_server_url, self.config.genrm_model
        )
        self._genrm_chat_completions_url = f"{normalized}/v1/chat/completions"
        return super().setup_webserver()

    async def _run_single_comparison(
        self,
        conversation_history: List[Dict[str, str]],
        response_obj_1: Dict[str, Any],
        response_obj_2: Dict[str, Any],
        pair_idx: Tuple[int, int] = (0, 0),
        principle: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """Run a single pairwise comparison via the externally-hosted GenRM model.

        Args:
            conversation_history: The conversation context
            response_obj_1: First Response API object
            response_obj_2: Second Response API object
            pair_idx: Tuple of (i, j) for logging
            principle: Optional principle for principle-based comparison

        Returns:
            Tuple of (score_1, score_2, ranking)
        """
        cfg = self.config

        # Extract final answer from Response API objects (GenRM only takes the final answer, not reasoning)
        response_1 = extract_output_text(response_obj_1)
        response_2 = extract_output_text(response_obj_2)

        # Format messages for GenRM using special roles 'response_1' and 'response_2'
        # The GenRM model's chat template handles these custom roles.
        # NOTE: NeMoGymEasyInputMessage validates role against the standard OpenAI
        # roles, so the custom-role messages are built with model_construct (no
        # validation). The Gym-managed genrm_model server does the equivalent via
        # GenRMConverter, which injects the custom roles server-side from metadata;
        # here the roles are injected client-side before POSTing chat completions.
        messages: List[NeMoGymEasyInputMessage] = [
            NeMoGymEasyInputMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                type="message",
            )
            for msg in conversation_history
        ]

        # Add principle message if enabled
        if cfg.use_principle:
            principle_text = principle if principle else cfg.default_principle
            messages.append(
                NeMoGymEasyInputMessage.model_construct(
                    role="principle", content=principle_text, type="message"
                )
            )

        messages.extend(
            [
                NeMoGymEasyInputMessage.model_construct(
                    role="response_1", content=response_1, type="message"
                ),
                NeMoGymEasyInputMessage.model_construct(
                    role="response_2", content=response_2, type="message"
                ),
            ]
        )

        try:
            # Retry logic for parse failures (not connection errors, which are handled by _post_chat_completions)
            max_attempts = max(1, int(cfg.genrm_parse_retries) + 1)

            for attempt_idx in range(max_attempts):
                payload = _build_chat_completions_payload(
                    cfg.genrm_responses_create_params, messages, cfg.genrm_model
                )
                try:
                    response_json = await _post_chat_completions(
                        "genrm_compare", self._genrm_chat_completions_url, payload,
                        max_retries=1, raise_on_context_length_error=True
                    )
                    genrm_answer = _extract_chat_completion_text(response_json)

                    score_1, score_2, ranking = parse_genrm_output(
                        genrm_answer,
                        cfg.default_score,
                        cfg.default_ranking,
                        raise_on_fail=True,
                    )
                    return score_1, score_2, ranking

                except GenRMOutputParseError:
                    if attempt_idx < max_attempts - 1:
                        await asyncio.sleep(float(cfg.genrm_parse_retry_sleep_s))
                        continue

                    logger.warning(
                        f"[GenRM] Parse failed for pair {pair_idx} after {max_attempts} attempts; "
                        f"falling back to defaults."
                    )
                    return cfg.default_score, cfg.default_score, cfg.default_ranking

                except BadRequestError as e:
                    logger.warning(
                        f"[GenRM] BadRequestError for pair {pair_idx}; falling back to defaults. "
                        f"Error: {e}"
                    )
                    return cfg.default_score, cfg.default_score, cfg.default_ranking

            return cfg.default_score, cfg.default_score, cfg.default_ranking

        except Exception as e:
            logger.error(f"[GenRM] Error in comparison for pair {pair_idx}: {e}")
            return cfg.default_score, cfg.default_score, cfg.default_ranking


if __name__ == "__main__":
    GenRMCompareResourcesServer.run_webserver()

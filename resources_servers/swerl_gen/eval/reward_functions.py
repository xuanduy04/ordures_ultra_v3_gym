
import logging
from typing import Dict


log = logging.getLogger(__name__)


def _calculate_patch_gen_reward(verification_result: Dict, scale_factor: float) -> float:
    """Map patch-generation sandbox resolution to a scalar reward."""
    if not verification_result:
        return 0.0

    score_map = {
        "RESOLVED_FULL": 1.0,
        "RESOLVED_PARTIAL": 0.2,
        "RESOLVED_NO": 0.0,
        "TIMEOUT": 0.0,
        "ERROR": 0.0,
    }

    status = verification_result.get("status")

    if status == "done":
        resolution = verification_result.get("resolution", "RESOLVED_NO")
        return score_map.get(resolution, 0.0) * scale_factor
    if status == "error":
        error_msg = verification_result.get("error", "Unknown error")
        log.debug("Verification ERROR: %s", error_msg)
        return score_map["ERROR"] * scale_factor
    if status == "timeout":
        log.debug("Verification TIMEOUT")
        return score_map["TIMEOUT"] * scale_factor

    log.debug("Unknown verification status: %s", status)
    return score_map["ERROR"] * scale_factor


def _calculate_test_gen_reward(verification_result: Dict, scale_factor: float) -> float:
    """Map reproduction-test sandbox result to a scalar reward."""
    if not verification_result:
        return 0.0

    score_map = {
        "TIMEOUT": 0.0,
        "ERROR": 0.0,
    }

    status = verification_result.get("status")

    if status == "done":
        return_codes_before_patch = verification_result.get("return_codes_before_patch", [])
        return_codes_after_patch = verification_result.get("return_codes_after_patch", [])
        if not return_codes_before_patch or not return_codes_after_patch:
            return 0.0 * scale_factor
        if (
            len(return_codes_before_patch) == 0
            or len(return_codes_after_patch) == 0
            or len(return_codes_before_patch) != len(return_codes_after_patch)
        ):
            return 0.0 * scale_factor
        if int(return_codes_before_patch[0]) == 2 and int(return_codes_after_patch[0]) == 0:
            log.debug(
                "Reproduction Test SUCCESS: %s -> %s",
                return_codes_before_patch,
                return_codes_after_patch,
            )
            return 1.0 * scale_factor
        if int(return_codes_before_patch[0]) == 1 or int(return_codes_after_patch[0]) == 1:
            log.debug(
                "Reproduction Test FAIL: %s -> %s",
                return_codes_before_patch,
                return_codes_after_patch,
            )
            return 0.0
        return 0.0 * scale_factor

    if status == "error":
        error_msg = verification_result.get("error", "Unknown error")
        log.debug("Verification ERROR: %s", error_msg)
        return score_map["ERROR"] * scale_factor
    if status == "timeout":
        log.debug("Verification TIMEOUT")
        return score_map["TIMEOUT"] * scale_factor

    log.debug("Unknown verification status: %s", status)
    return score_map["ERROR"] * scale_factor

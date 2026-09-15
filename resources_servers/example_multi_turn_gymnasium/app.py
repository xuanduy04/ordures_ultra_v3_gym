

"""Scripted multi-turn example. Replays follow-up messages from verifier_metadata.

verifier_metadata fields:
  follow_ups:      List[str]   follow-up messages sent after each assistant turn
  expected_answer: str         substring expected in the final assistant response
"""

from typing import Dict, Optional

from pydantic import Field

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.gymnasium import GymnasiumServer, extract_text


class ExampleMultiTurnEnv(GymnasiumServer):
    session_turns: Dict[str, int] = Field(default_factory=dict)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        """Returns (observation, info)."""
        self.session_turns[session_id] = 0
        return None, {}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        """Returns (observation, reward, terminated, truncated, info)."""
        follow_ups = metadata.get("follow_ups", [])
        turn = self.session_turns.get(session_id, 0)

        if turn < len(follow_ups):
            self.session_turns[session_id] = turn + 1
            return follow_ups[turn], 0.0, False, False, {}

        expected = metadata.get("expected_answer", "")
        text = extract_text(action)
        reward = 1.0 if expected and expected.lower() in text.lower() else 0.0
        return None, reward, True, False, {}


if __name__ == "__main__":
    ExampleMultiTurnEnv.run_webserver()

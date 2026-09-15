
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.vlm_eval_kit.app import VlmEvalKitResourcesServer, VlmEvalKitResourcesServerConfig


class TestApp:
    def test_sanity(self) -> None:
        config = VlmEvalKitResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        VlmEvalKitResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

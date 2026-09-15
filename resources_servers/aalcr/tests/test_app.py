
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.aalcr.app import AalcrResourcesServer, AalcrResourcesServerConfig


class TestApp:
    def test_sanity(self) -> None:
        config = AalcrResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server={
                "type": "responses_api_models",
                "name": "abcd",
            },
            judge_responses_create_params_overrides=dict(),
        )
        AalcrResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

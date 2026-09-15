
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.example_multi_turn_gymnasium.app import ExampleMultiTurnEnv


class TestApp:
    def test_sanity(self) -> None:
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
        ExampleMultiTurnEnv(config=config, server_client=MagicMock(spec=ServerClient))

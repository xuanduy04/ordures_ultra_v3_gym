
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.ruler.app import RulerResourcesServer, RulerResourcesServerConfig


class TestApp:
    def test_sanity(self) -> None:
        config = RulerResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        RulerResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.example_single_tool_call.app import (
    SimpleWeatherResourcesServer,
    SimpleWeatherResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleWeatherResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        SimpleWeatherResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

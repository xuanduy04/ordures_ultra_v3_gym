
from nemo_gym.server_utils import ServerClient

from app import ExampleMultiStepResourcesServer, ExampleMultiStepResourcesServerConfig

from unittest.mock import MagicMock


class TestApp:
    def test_sanity(self) -> None:
        config = ExampleMultiStepResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        ExampleMultiStepResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

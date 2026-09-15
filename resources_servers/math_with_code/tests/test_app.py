
from unittest.mock import MagicMock

from app import PythonExecutorResourcesServer, PythonExecutorResourcesServerConfig

from nemo_gym.server_utils import ServerClient


class TestApp:
    """Tests for the Python Executor server."""

    SERVER_NAME = "math_with_code"

    def test_sanity(self) -> None:
        """Basic instantiation test - always runs."""
        config = PythonExecutorResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        PythonExecutorResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

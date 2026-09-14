
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.arc_agi.app import (
    ARCAGIResourcesServer,
    ARCAGIResourcesServerConfig,
    _parse_grid,
)


class TestApp:
    def test_sanity(self) -> None:
        config = ARCAGIResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="test_arc_agi",
        )
        ARCAGIResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_parse_grid(self) -> None:
        grid_text = "[[1,2],[3,4]]"
        result = _parse_grid(grid_text)
        assert result == [[1, 2], [3, 4]]

        grid_text = "[[1,2,3],[4,5,6],[7,8,9]]"
        result = _parse_grid(grid_text)
        assert result == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        grid_text = "[1,2,3,4"
        result = _parse_grid(grid_text)
        assert result is None

        result = _parse_grid("")
        assert result is None

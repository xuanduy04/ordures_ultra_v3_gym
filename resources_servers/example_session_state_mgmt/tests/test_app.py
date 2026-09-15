
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from httpx import Cookies

from nemo_gym.server_utils import ServerClient
from resources_servers.example_session_state_mgmt.app import (
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = StatefulCounterResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        server = StatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        app = server.setup_webserver()
        client = TestClient(app)

        class StatelessCookies(Cookies):
            def extract_cookies(self, response):
                pass

        client._cookies = StatelessCookies(client._cookies)

        # Check that we are at 0
        response = client.post("/get_counter_value")
        initial_request_cookies = response.cookies
        assert response.json() == {"count": 0}
        response = client.post("/increment_counter", json={"count": 2}, cookies=initial_request_cookies)
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", cookies=initial_request_cookies)
        assert response.json() == {"count": 2}

        # Start a new session i.e. don't pass cookies
        response = client.post("/increment_counter", json={"count": 4})
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", cookies=response.cookies)
        assert response.json() == {"count": 4}
        response = client.post("/increment_counter", json={"count": 3}, cookies=response.cookies)
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", cookies=response.cookies)
        assert response.json() == {"count": 7}

        response = client.post("/get_counter_value", cookies=initial_request_cookies)
        assert response.json() == {"count": 2}

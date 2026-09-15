

import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.utils_outsource import judge_server_url_utils as jsu
from resources_servers.utils_outsource.judge_server_url_utils import (
    _build_chat_completions_payload,
    _extract_chat_completion_text,
    _messages_to_chat_format,
    _normalize_judge_server_url,
    _post_chat_completions,
)


class TestNormalizeJudgeServerUrl:
    def test_bare_host_port(self):
        assert _normalize_judge_server_url("0.0.0.0:8000") == "http://0.0.0.0:8000"

    def test_full_url(self):
        assert _normalize_judge_server_url("http://0.0.0.0:8000") == "http://0.0.0.0:8000"

    def test_drops_path(self):
        assert _normalize_judge_server_url("http://0.0.0.0:8000/extra/stuff/behind") == "http://0.0.0.0:8000"

    def test_https(self):
        assert _normalize_judge_server_url("https://judge.example.com:8443") == "https://judge.example.com:8443"

    def test_strips_whitespace_and_slashes(self):
        assert _normalize_judge_server_url("  0.0.0.0:8000/  ") == "http://0.0.0.0:8000"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _normalize_judge_server_url("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            _normalize_judge_server_url("   ")

    def test_env_name_in_error(self):
        with pytest.raises(ValueError, match="test_env"):
            _normalize_judge_server_url("", env_name="test_env")


class TestMessagesToChatFormat:
    def test_string_content(self):
        msgs = [
            NeMoGymEasyInputMessage(role="system", content="be precise"),
            NeMoGymEasyInputMessage(role="user", content="judge this"),
        ]
        out = _messages_to_chat_format(msgs)
        assert out == [
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "judge this"},
        ]

    def test_list_content_text_parts(self):
        msgs = [
            NeMoGymEasyInputMessage(
                role="user",
                content=[
                    {"type": "input_text", "text": "hello "},
                    {"type": "input_text", "text": "world"},
                ],
            ),
        ]
        out = _messages_to_chat_format(msgs)
        assert out == [{"role": "user", "content": "hello world"}]


class TestBuildChatCompletionsPayload:
    def test_maps_max_output_tokens_to_max_tokens(self):
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=8192, temperature=0.7, top_p=0.8
        )
        msgs = [NeMoGymEasyInputMessage(role="user", content="q")]
        payload = _build_chat_completions_payload(params, msgs, "my-model")
        assert payload["model"] == "my-model"
        assert payload["stream"] is False
        assert payload["max_tokens"] == 8192
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.8
        assert payload["messages"] == [{"role": "user", "content": "q"}]

    def test_omits_optional_when_none(self):
        params = NeMoGymResponseCreateParamsNonStreaming(input=[])
        msgs = [NeMoGymEasyInputMessage(role="user", content="q")]
        payload = _build_chat_completions_payload(params, msgs, "m")
        assert "max_tokens" not in payload
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert payload["model"] == "m"


class TestExtractChatCompletionText:
    def test_string_content(self):
        data = {"choices": [{"message": {"content": "Analysis... [[YES]]"}}]}
        assert _extract_chat_completion_text(data) == "Analysis... [[YES]]"

    def test_list_content(self):
        data = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
        assert _extract_chat_completion_text(data) == "ab"

    def test_no_choices(self):
        assert _extract_chat_completion_text({}) == ""
        assert _extract_chat_completion_text({"choices": []}) == ""


@pytest.fixture
async def client_session(monkeypatch):
    session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(force_close=True), timeout=aiohttp.ClientTimeout()
    )
    monkeypatch.setattr(jsu, "get_global_aiohttp_client", lambda: session)
    yield session
    await session.close()


@pytest.fixture
async def fake_judge():
    servers = []

    async def start(handler):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        return str(server.make_url("/v1/chat/completions"))

    yield start

    for server in servers:
        await server.close()


class TestPostChatCompletions:
    """Transport behavior of _post_chat_completions: connection retries, semaphore, no total timeout."""

    def _payload(self):
        return {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}

    async def test_connection_drops_retried_beyond_max_retries(self, client_session, fake_judge, monkeypatch):
        """A disconnect must not consume the bounded attempt budget: max_retries=1 still succeeds."""
        monkeypatch.setattr(jsu, "_get_retry_delay", lambda attempt: 0.0)
        drops = {"count": 0}

        async def handler(request):
            if drops["count"] < 2:
                drops["count"] += 1
                request.transport.abort()
                return web.Response()
            return web.json_response({"choices": [{"message": {"content": "ok"}}]})

        url = await fake_judge(handler)
        result = await _post_chat_completions("test_env", url, self._payload(), max_retries=1)

        assert drops["count"] == 2
        assert result["choices"][0]["message"]["content"] == "ok"

    async def test_semaphore_caps_simultaneous_requests(self, client_session, fake_judge):
        state = {"inflight": 0, "max_inflight": 0}

        async def handler(request):
            state["inflight"] += 1
            state["max_inflight"] = max(state["max_inflight"], state["inflight"])
            await asyncio.sleep(0.05)
            state["inflight"] -= 1
            return web.json_response({"choices": [{"message": {"content": "ok"}}]})

        url = await fake_judge(handler)
        semaphore = asyncio.Semaphore(2)
        results = await asyncio.gather(
            *[_post_chat_completions("test_env", url, self._payload(), semaphore=semaphore) for _ in range(6)]
        )

        assert all(r["choices"][0]["message"]["content"] == "ok" for r in results)
        assert state["max_inflight"] <= 2

    async def test_no_total_timeout_and_connection_close(self, client_session, fake_judge, monkeypatch):
        captured = {}
        real_timeout = aiohttp.ClientTimeout

        def spy_timeout(**kwargs):
            captured.update(kwargs)
            return real_timeout(**kwargs)

        monkeypatch.setattr(jsu, "ClientTimeout", spy_timeout)
        seen_headers = {}

        async def handler(request):
            seen_headers.update(request.headers)
            return web.json_response({"choices": [{"message": {"content": "ok"}}]})

        url = await fake_judge(handler)
        result = await _post_chat_completions("test_env", url, self._payload())

        assert result["choices"][0]["message"]["content"] == "ok"
        assert captured["total"] is None
        assert captured["sock_read"] == jsu._REQUEST_TIMEOUT_SECONDS
        assert seen_headers["Connection"] == "close"

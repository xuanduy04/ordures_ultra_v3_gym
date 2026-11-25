import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response


pytest.importorskip("nemo_gym", reason="Gym adapter tests run in the existing Gym runtime")

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.spreadsheet_agent.app import SpreadsheetAgent, SpreadsheetAgentConfig


def reply(value, status=200):
    encoded = json.dumps(value).encode()
    return SimpleNamespace(
        status=status,
        ok=status < 400,
        read=AsyncMock(return_value=encoded),
        content=SimpleNamespace(read=AsyncMock(return_value=encoded)),
        cookies={},
    )


def model_response(output):
    return {
        "id": "r",
        "created_at": 0,
        "model": "test-policy",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
    }


def final_message():
    return {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Done", "annotations": []}],
    }


def agent():
    client = MagicMock(spec=ServerClient)
    config = SpreadsheetAgentConfig(
        host="127.0.0.1",
        port=8001,
        name="spreadsheet_agent",
        entrypoint="app.py",
        max_steps=3,
        resources_server={"type": "resources_servers", "name": "resources"},
        model_server={"type": "responses_api_models", "name": "policy"},
    )
    return SpreadsheetAgent(config=config, server_client=client)


def request():
    return Request({"type": "http", "headers": [(b"cookie", b"spreadsheet_test=signed-session-cookie")]})


@pytest.mark.asyncio
async def test_agent_propagates_call_id_cookies_and_training_tokens():
    server = agent()
    tool = {
        "type": "function_call",
        "name": "write_range",
        "call_id": "provider-call-42",
        "arguments": '{"range":"A1","data":42}',
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3, 4],
        "generation_log_probs": [-0.1, -0.2],
    }
    server.server_client.post = AsyncMock(
        side_effect=[
            reply(model_response([tool])),
            reply({"changed": True}),
            reply(model_response([final_message()])),
        ]
    )
    result = await server._responses_once(request(), Response(), NeMoGymResponseCreateParamsNonStreaming(input=[]))
    calls = server.server_client.post.call_args_list
    assert calls[1].kwargs["headers"] == {"X-Spreadsheet-Operation-Id": "1.0.provider-call-42"}
    assert calls[1].kwargs["cookies"]["spreadsheet_test"] == "signed-session-cookie"
    assert result.output[0].model_dump()["generation_token_ids"] == [3, 4]
    assert calls[2].kwargs["json"].input[0].model_dump()["generation_log_probs"] == [-0.1, -0.2]


@pytest.mark.asyncio
async def test_model_cannot_invoke_lifecycle_or_verifier_endpoints():
    server = agent()
    attack = {"type": "function_call", "name": "verify", "call_id": "c", "arguments": "{}"}
    server.server_client.post = AsyncMock(
        side_effect=[
            reply(model_response([attack])),
            reply(model_response([final_message()])),
        ]
    )
    result = await server._responses_once(request(), Response(), NeMoGymResponseCreateParamsNonStreaming(input=[]))
    assert all(c.kwargs["server_name"] == "policy" for c in server.server_client.post.call_args_list)
    assert "Unknown spreadsheet tool" in result.output[1].output


@pytest.mark.asyncio
async def test_retried_responses_share_one_agent_execution():
    server = agent()
    server.server_client.post = AsyncMock(
        side_effect=[reply({"claimed": True}), reply(model_response([final_message()]))]
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input=[])
    a, b = await asyncio.gather(
        server.responses(request(), Response(), body), server.responses(request(), Response(), body)
    )
    assert a.model_dump() == b.model_dump()
    paths = [c.kwargs["url_path"] for c in server.server_client.post.call_args_list]
    assert paths == ["/begin_execution", "/v1/responses"]


def queued_request(index):
    return Request({"type": "http", "headers": [(b"cookie", f"spreadsheet_test=session-{index}".encode())]})


@pytest.mark.asyncio
async def test_4000_pending_responses_are_bounded_and_completed_cache_does_not_block_admission(monkeypatch):
    from fastapi import HTTPException
    from nemo_gym.openai_utils import NeMoGymResponse

    server = agent()
    release = asyncio.Event()
    busy, peak = 0, 0

    async def perform(self, request, response, body):
        nonlocal busy, peak
        busy += 1
        peak = max(peak, busy)
        try:
            await release.wait()
            return NeMoGymResponse.model_validate(model_response([final_message()]))
        finally:
            busy -= 1

    monkeypatch.setattr(SpreadsheetAgent, "_responses_once", perform)
    server.server_client.post = AsyncMock(return_value=reply({"claimed": True}))
    body = NeMoGymResponseCreateParamsNonStreaming(input=[])
    callers = [asyncio.create_task(server.responses(queued_request(i), Response(), body)) for i in range(4000)]
    duplicate = asyncio.create_task(server.responses(queued_request(0), Response(), body))
    try:
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(server.response_tasks) == 4000
        assert peak == server.config.max_concurrent_responses
        with pytest.raises(HTTPException, match="pending response queue is full") as error:
            await server.responses(queued_request(4000), Response(), body)
        assert error.value.status_code == 503
    finally:
        release.set()
        results = await asyncio.gather(*callers, duplicate)
    assert server.server_client.post.await_count == 4000
    assert results[0].model_dump() == results[-1].model_dump()
    assert busy == 0
    assert len(server.response_completed_at) == 4000

    # A full completed-result cache must not reject a new, distinct rollout.
    await server.responses(queued_request(4000), Response(), body)
    assert server.server_client.post.await_count == 4001
    assert len(server.response_completed_at) == 4000
    assert len(server.response_tasks) == 4000
    await server.responses(queued_request(4000), Response(), body)
    assert server.server_client.post.await_count == 4001


@pytest.mark.asyncio
async def test_cancelled_http_caller_does_not_replay_or_release_inflight_execution(monkeypatch):
    from nemo_gym.openai_utils import NeMoGymResponse

    server = agent()
    release, entered = asyncio.Event(), asyncio.Event()

    async def perform(self, request, response, body):
        entered.set()
        await release.wait()
        return NeMoGymResponse.model_validate(model_response([final_message()]))

    monkeypatch.setattr(SpreadsheetAgent, "_responses_once", perform)
    server.server_client.post = AsyncMock(return_value=reply({"claimed": True}))
    body = NeMoGymResponseCreateParamsNonStreaming(input=[])
    caller = asyncio.create_task(server.responses(queued_request(1), Response(), body))
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert not next(iter(server.response_tasks.values()))[1].done()
    retry = asyncio.create_task(server.responses(queued_request(1), Response(), body))
    release.set()
    await retry
    assert server.server_client.post.await_count == 1


@pytest.mark.asyncio
async def test_completed_response_ttl_starts_when_execution_finishes(monkeypatch):
    from nemo_gym.openai_utils import NeMoGymResponse

    server = agent()
    clock = [10.0]
    monkeypatch.setattr("responses_api_agents.spreadsheet_agent.app.time.monotonic", lambda: clock[0])

    async def perform(self, request, response, body):
        clock[0] += 500
        return NeMoGymResponse.model_validate(model_response([final_message()]))

    monkeypatch.setattr(SpreadsheetAgent, "_responses_once", perform)
    server.server_client.post = AsyncMock(return_value=reply({"claimed": True}))
    body = NeMoGymResponseCreateParamsNonStreaming(input=[])
    await server.responses(queued_request(1), Response(), body)
    await server.responses(queued_request(1), Response(), body)
    assert server.server_client.post.await_count == 1
    clock[0] += 301
    server._prune_responses()
    assert not server.response_tasks


@pytest.mark.asyncio
async def test_multiple_tool_turns_keep_cookies_when_a_response_has_no_set_cookie():
    server = agent()
    first = {"type": "function_call", "name": "list_sheets", "call_id": "read-1", "arguments": "{}"}
    second = {"type": "function_call", "name": "inspect_range", "call_id": "read-2", "arguments": '{"range":"A1"}'}
    server.server_client.post = AsyncMock(
        side_effect=[
            reply(model_response([first])),
            reply({"sheets": []}),
            reply(model_response([second])),
            reply({"cells": []}),
            reply(model_response([final_message()])),
        ]
    )
    await server._responses_once(request(), Response(), NeMoGymResponseCreateParamsNonStreaming(input=[]))
    calls = server.server_client.post.call_args_list
    assert calls[3].kwargs["cookies"]["spreadsheet_test"] == "signed-session-cookie"
    assert calls[3].kwargs["headers"]["X-Spreadsheet-Operation-Id"] == "2.0.read-2"


def test_adapter_refreshes_cookie_on_read_only_tool(monkeypatch):
    from fastapi.testclient import TestClient

    from resources_servers.spreadsheet.app import SpreadsheetConfig, SpreadsheetResourcesServer

    config = SpreadsheetConfig(
        host="127.0.0.1",
        port=8003,
        entrypoint="app.py",
        name="test",
        service_url="http://service:8088",
        api_key="x" * 40,
        session_secret_key="y" * 40,
    )
    server = SpreadsheetResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    backend = AsyncMock(
        side_effect=[
            {"session_id": "a" * 32},
            {"job_id": "j", "status": "DONE", "result": {"ok": True, "output": {"sheets": []}}},
        ]
    )
    monkeypatch.setattr(SpreadsheetResourcesServer, "_call", backend)
    client = TestClient(server.setup_webserver())
    seeded = client.post("/seed_session", json={"task_id": "task", "rollout_id": "rollout"})
    assert seeded.status_code == 200
    response = client.post("/list_sheets", json={}, headers={"X-Spreadsheet-Operation-Id": "read"})
    assert response.status_code == 200
    assert "spreadsheet_test" in response.cookies


def adapter(monkeypatch, backend):
    from fastapi.testclient import TestClient

    from resources_servers.spreadsheet.app import SpreadsheetConfig, SpreadsheetResourcesServer

    config = SpreadsheetConfig(
        host="127.0.0.1",
        port=8003,
        entrypoint="app.py",
        name="test",
        service_url="http://service:8088",
        api_key="x" * 40,
        session_secret_key="y" * 40,
    )
    server = SpreadsheetResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(SpreadsheetResourcesServer, "_call", backend)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json={"task_id": "task", "rollout_id": "rollout"}).status_code == 200
    return client


def test_oversized_model_arguments_are_a_tool_observation_not_an_outage(monkeypatch):
    backend = AsyncMock(side_effect=[{"session_id": "a" * 32}])
    client = adapter(monkeypatch, backend)
    # 25k Vietnamese characters is within the 65,536-character schema limit but
    # exceeds the service's escaped-JSON argument limit.
    response = client.post(
        "/code_interpreter", json={"code": "# " + "ệ" * 25_000}, headers={"X-Spreadsheet-Operation-Id": "c"}
    )
    assert response.status_code == 200
    assert response.json() == {"ok": False, "error_type": "tool_error", "error": "Tool arguments exceed the size limit"}
    assert backend.await_count == 1


def test_service_tool_rejection_is_returned_to_the_model(monkeypatch):
    from resources_servers.spreadsheet import app as adapter_module
    from resources_servers.spreadsheet.app import SpreadsheetConfig, SpreadsheetResourcesServer

    def http(status, value):
        return SimpleNamespace(status=status, read=AsyncMock(return_value=json.dumps(value).encode()), release=MagicMock())

    async def fake_request(method, url, **kwargs):
        if url.endswith("/sessions"):
            return http(200, {"session_id": "a" * 32})
        return http(400, {"detail": "Unknown tool", "error_type": "tool_error"})

    monkeypatch.setattr(adapter_module, "gym_request", fake_request)
    from fastapi.testclient import TestClient

    config = SpreadsheetConfig(
        host="127.0.0.1",
        port=8003,
        entrypoint="app.py",
        name="test",
        service_url="http://service:8088",
        api_key="x" * 40,
        session_secret_key="y" * 40,
    )
    server = SpreadsheetResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json={"task_id": "task", "rollout_id": "rollout"}).status_code == 200
    response = client.post("/list_sheets", json={}, headers={"X-Spreadsheet-Operation-Id": "c"})
    assert response.status_code == 200
    assert response.json() == {"ok": False, "error_type": "tool_error", "error": "Unknown tool"}


def verify_body():
    return {
        "task_id": "task",
        "responses_create_params": {"input": []},
        "response": model_response([final_message()]),
    }


def test_unscorable_final_workbook_scores_zero(monkeypatch):
    failed = {"ok": False, "error_type": "tool_error", "error": "LibreOffice worker timed out; changes discarded"}
    backend = AsyncMock(side_effect=[{"session_id": "a" * 32}, {"job_id": "j", "status": "DONE", "result": failed}])
    client = adapter(monkeypatch, backend)
    response = client.post("/verify", json=verify_body())
    assert response.status_code == 200
    result = response.json()
    assert result["reward"] == 0.0 and result["cell_accuracy"] == 0.0
    assert result["engine"] is None and result["verify_error"] == failed["error"]


def test_environment_failure_during_verify_still_aborts(monkeypatch):
    failed = {"ok": False, "error_type": "environment_error", "error": "Execution backend failed"}
    backend = AsyncMock(side_effect=[{"session_id": "a" * 32}, {"job_id": "j", "status": "DONE", "result": failed}])
    client = adapter(monkeypatch, backend)
    assert client.post("/verify", json=verify_body()).status_code == 503


@pytest.mark.asyncio
async def test_unencodable_model_arguments_are_tool_errors_not_failed_rollouts():
    server = agent()
    calls = [
        {"type": "function_call", "name": "write_range", "call_id": "s", "arguments": '{"range":"A1","data":"\\ud800"}'},
        {"type": "function_call", "name": "write_range", "call_id": "b", "arguments": '{"range":"A1","data":1' + "0" * 30 + "}"},
    ]
    server.server_client.post = AsyncMock(
        side_effect=[reply(model_response(calls)), reply(model_response([final_message()]))]
    )
    result = await server._responses_once(request(), Response(), NeMoGymResponseCreateParamsNonStreaming(input=[]))
    assert all(c.kwargs["server_name"] == "policy" for c in server.server_client.post.call_args_list)
    assert all("Invalid tool call arguments" in item.output for item in result.output[2:4])


@pytest.mark.asyncio
async def test_repeated_provider_call_ids_get_distinct_operation_ids():
    server = agent()
    # Per-turn provider IDs, as produced by e.g. kimi_k2-style tool parsers.
    repeated = {"type": "function_call", "name": "list_sheets", "call_id": "functions.list_sheets:0", "arguments": "{}"}
    server.server_client.post = AsyncMock(
        side_effect=[
            reply(model_response([repeated])),
            reply({"sheets": []}),
            reply(model_response([repeated])),
            reply({"sheets": []}),
            reply(model_response([final_message()])),
        ]
    )
    await server._responses_once(request(), Response(), NeMoGymResponseCreateParamsNonStreaming(input=[]))
    headers = [c.kwargs["headers"]["X-Spreadsheet-Operation-Id"] for c in server.server_client.post.call_args_list if "headers" in c.kwargs]
    assert headers == ["1.0.functions.list_sheets:0", "2.0.functions.list_sheets:0"]

import sqlite3
import sys
import time
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from xml.etree.ElementTree import ParseError

import pytest

from resources_servers.spreadsheet.service import engines, ooxml_metadata, workbook
from resources_servers.spreadsheet.service.errors import EnvironmentError, ToolError
from resources_servers.spreadsheet.service.state import Store


# These regressions use test doubles only: no workbook files are created.


def test_cell_values_and_formula_objects_stay_json_values():
    array_formula = SimpleNamespace(ref="A1:A3", text="=SUM(B1:B3*C1:C3)")

    class DataTableFormula:
        ref = "B2:C4"

    assert workbook.json_value(timedelta(days=1, hours=2)) == "1 day, 2:00:00"
    assert workbook.json_value(float("nan")) == "nan"
    assert workbook.json_value(True) is True
    assert workbook.formula_text("=A1") == "=A1"
    assert workbook.formula_text(array_formula) == "=SUM(B1:B3*C1:C3)"
    assert workbook.formula_text(DataTableFormula()) == "DataTableFormula"


def test_workbook_parser_failures_are_tool_errors(monkeypatch):
    fake = ModuleType("openpyxl")

    def load_workbook(*args, **kwargs):
        raise ParseError("not well-formed (invalid token)")

    fake.load_workbook = load_workbook
    monkeypatch.setitem(sys.modules, "openpyxl", fake)
    monkeypatch.setattr(workbook, "check_archive", lambda path: None)
    with pytest.raises(ToolError, match="cannot be opened"):
        workbook.load("data.xlsx")


class Book:
    epoch = None

    def close(self):
        pass


def test_reward_separates_task_defects_from_unreadable_model_workbooks(monkeypatch):
    def load(path, **kwargs):
        if path == "broken":
            raise ToolError("Workbook cannot be opened")
        return Book()

    monkeypatch.setattr(workbook, "load", load)
    with pytest.raises(EnvironmentError, match="oracle"):
        workbook.compare("broken", "actual", "A1")

    def invalid_metadata(book, position):
        raise ToolError("Worksheet does not exist")

    monkeypatch.setattr(workbook, "answer_regions", invalid_metadata)
    with pytest.raises(EnvironmentError, match="answer metadata"):
        workbook.compare("oracle", "actual", "Missing!A1")
    monkeypatch.setattr(workbook, "answer_regions", lambda book, position: [])
    with pytest.raises(ToolError, match="cannot be opened"):
        workbook.compare("oracle", "broken", "A1")


class Process:
    pid = 1

    def __init__(self):
        self.stdin = SimpleNamespace(write=lambda text: None, flush=lambda: None)


@pytest.fixture
def pool(tmp_path, monkeypatch):
    pool = engines.LibreOfficePool(tmp_path / "office", workers=1, timeout=5)
    killed = []
    monkeypatch.setattr(engines, "terminate", killed.append)
    monkeypatch.setattr(ooxml_metadata, "capture", lambda path: {})
    monkeypatch.setattr(ooxml_metadata, "restore", lambda path, metadata: [])
    monkeypatch.setattr(workbook, "load", lambda path, **kwargs: Book())
    pool.process = Process()
    pool._get_process = lambda slot, deadline: pool.process
    pool.killed = killed
    return pool


def test_libreoffice_failure_on_a_workbook_is_a_tool_error_and_restarts_the_worker(pool):
    def fail(process, deadline):
        raise EnvironmentError("LibreOffice could not recalculate this workbook")

    pool._read = fail
    with pytest.raises(ToolError, match="changes discarded"):
        pool.recalculate(SimpleNamespace(resolve=lambda: "data.xlsx"))
    assert pool.killed == [pool.process]
    assert pool.slots.qsize() == 1


def test_libreoffice_startup_failure_stays_an_environment_error(pool):
    def unavailable(slot, deadline):
        raise EnvironmentError("LibreOffice startup failed")

    pool._get_process = unavailable
    with pytest.raises(EnvironmentError, match="startup"):
        pool.recalculate(SimpleNamespace(resolve=lambda: "data.xlsx"))
    assert pool.slots.qsize() == 1


def test_unparseable_or_unreadable_workbooks_are_tool_errors(pool, monkeypatch):
    pool._read = lambda process, deadline: {"ok": True}
    assert pool.recalculate(SimpleNamespace(resolve=lambda: "data.xlsx")) == {"metadata_repairs": []}

    def unparseable(path):
        raise ParseError("not well-formed (invalid token)")

    monkeypatch.setattr(ooxml_metadata, "capture", unparseable)
    with pytest.raises(ToolError, match="cannot be parsed"):
        pool.recalculate(SimpleNamespace(resolve=lambda: "data.xlsx"))
    monkeypatch.setattr(ooxml_metadata, "capture", lambda path: {})

    def unreadable(path, **kwargs):
        raise ToolError("Workbook cannot be opened")

    monkeypatch.setattr(workbook, "load", unreadable)
    with pytest.raises(ToolError, match="cannot be opened"):
        pool.recalculate(SimpleNamespace(resolve=lambda: "data.xlsx"))


class Engine:
    version = "test-double"

    def close(self):
        pass


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "sessions", SimpleNamespace(manifest_sha256="x"), Engine(), None, workers=1)
    with sqlite3.connect(store.db_path) as db:
        db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)", ("a" * 32, "r", "t", "OPEN", time.time(), time.time()))
    yield store
    store.shutdown()


def finished(store, receipt):
    deadline = time.monotonic() + 5
    while (job := store.job(receipt["job_id"]))["status"] != "DONE":
        assert time.monotonic() < deadline, "job stayed " + job["status"]
        time.sleep(0.01)
    return job["result"]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"duration": timedelta(hours=1)}, {"ok": True, "output": {"duration": "1:00:00"}}),
        ({"value": float("nan")}, {"ok": False, "error_type": "environment_error"}),
    ],
)
def test_jobs_always_finish_even_with_unserializable_output(store, monkeypatch, output, expected):
    monkeypatch.setattr(store, "_execute", lambda session, action, arguments, job_id: {"ok": True, "output": output})
    result = finished(store, store.submit("a" * 32, "call", "list_sheets", {}))
    assert {key: result[key] for key in expected} == expected


def test_oversized_and_non_finite_arguments_are_tool_errors(store):
    with pytest.raises(ToolError, match="size limit"):
        store.submit("a" * 32, "big", "code_interpreter", {"code": "ệ" * 25_000})
    with pytest.raises(ToolError, match="finite"):
        store.submit("a" * 32, "nan", "write_range", {"range": "A1", "data": float("nan")})


def test_pending_job_count_uses_an_index(store):
    with store.db() as db:
        plan = db.execute(
            "EXPLAIN QUERY PLAN SELECT count(*) FROM jobs WHERE status IN ('QUEUED','RUNNING')"
        ).fetchall()
    assert any("jobs_status" in str(tuple(row)) for row in plan)

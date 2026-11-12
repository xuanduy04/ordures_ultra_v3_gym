"""Minimal HTTP calculator service for agent tool calls."""

from __future__ import annotations

import ast
import json
import math
import operator
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_FUNCTIONS = {
    "exp": math.exp,
    "log": math.log,
}


def evaluate(expression: str) -> int | float:
    """Evaluate a basic arithmetic expression without executing arbitrary code."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    if len(expression) > 1_000:
        raise ValueError("expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid arithmetic expression") from exc

    value = _evaluate_node(tree.body)
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError("result is too large") from exc
    if not finite:
        raise ValueError("result is not finite")
    return value


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("boolean values are not supported")
        return node.value

    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        try:
            return _BINARY_OPERATORS[type(node.op)](left, right)
        except (ArithmeticError, OverflowError) as exc:
            raise ValueError("arithmetic operation failed") from exc

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in _FUNCTIONS and not node.keywords:
            arguments = [_evaluate_node(argument) for argument in node.args]
            try:
                return _FUNCTIONS[node.func.id](*arguments)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError("invalid function arguments or result") from exc

    raise ValueError("only arithmetic expressions are supported")


class CalculatorHandler(BaseHTTPRequestHandler):
    server_version = "CalculatorTool/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/calculate":
            self._send_json(404, {"error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > 1_000_000:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(content_length))
            result = evaluate(payload["expression"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        self._send_json(200, {"expression": payload["expression"], "result": result})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class CalculatorServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    server = CalculatorServer((host, port), CalculatorHandler)
    print(f"Calculator tool listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

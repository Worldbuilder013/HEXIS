"""Execute FSM tool calls through OpenCode's native session/tool loop.

A loopback OpenAI-compatible provider emits exactly the requested tool call.
It performs no inference. OpenCode validates and executes the native tool, then
receives a stop completion. One server/session is retained for the whole run.
API reference: https://opencode.ai/docs/server/
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from hexis.tools.backends import registry as _registry
from hexis.tools.toolspec import ToolSpec

#: native tools this backend can execute: taken from the registry backends/opencode.json.
REGISTRY = _registry("opencode")
PRIMITIVES = tuple(REGISTRY)

FALLBACK_PROTOCOL = '''Return exactly one JSON action per step.
Tool action: {"kind":"tool","name":"bash|read|write|edit|list|glob|grep",
"input":{native tool arguments},"writes":["ok","returncode","stdout","stderr"]}.
Use OpenCode native arguments: bash(command, description), read(filePath),
write(filePath, content), edit(filePath, oldString, newString),
list(path), glob(pattern, path), grep(pattern, path, include).
Never use file_ops. Tool results contain ok, returncode, stdout, stderr.
End action: {"kind":"end","terminal":"done"}.
Follow the skill document below.\n\n'''


class OpenCodeError(RuntimeError):
    pass


def tool_result(part: dict, name: str, args: dict) -> dict:
    state = part.get("state") or {}
    if part.get("tool") != name or state.get("input") != args:
        raise OpenCodeError("OpenCode executed a different tool or different arguments")
    status = state.get("status")
    if status not in ("completed", "error"):
        raise OpenCodeError(f"OpenCode tool did not finish: {status}")
    metadata = state.get("metadata") or {}
    rc = int(metadata.get("exit") or 0) if status == "completed" else 1
    return {"ok": status == "completed" and rc == 0, "returncode": rc,
            "stdout": str(state.get("output") or "") if status == "completed" else "",
            "stderr": str(state.get("error") or "") if status == "error" else "",
            "metadata": metadata, "opencode_part": deepcopy(part)}


class OpenCodeTools:
    def __init__(self, workdir: Path, *, binary: str = "opencode", timeout_s: float = 120):
        self.workdir = Path(workdir).absolute()
        self.binary = shutil.which(binary)
        if not self.binary:
            raise OpenCodeError(f"OpenCode executable not found: {binary}")
        self.timeout_s = timeout_s
        self.calls = []
        self.pending = None
        self.lock = threading.Lock()
        self.process = self.server = self.client = self.temp = None
        self.provider_error = ""
        self.secret = secrets.token_urlsafe(32)

    def __enter__(self):
        owner = self

        class Provider(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                if self.headers.get("Authorization") != "Bearer " + owner.secret:
                    self.send_error(401)
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    pending = owner.pending
                    if pending is None or pending.get("sent"):
                        message = {"role": "assistant", "content": "Done."}
                        finish = "stop"
                    elif any(m.get("role") == "tool" and m.get("tool_call_id") == pending["id"]
                             for m in body.get("messages", [])):
                        message = {"role": "assistant", "content": "Done."}
                        finish = "stop"
                    else:
                        names = {t.get("function", {}).get("name") for t in body.get("tools", [])}
                        if pending["name"] not in names:
                            owner.provider_error = f"OpenCode does not expose native tool {pending['name']!r}"
                            message = {"role": "assistant", "content": "Tool unavailable."}
                            finish = "stop"
                        else:
                            pending["sent"] = True
                            message = {"role": "assistant", "content": None, "tool_calls": [{
                                "id": pending["id"], "type": "function", "function": {
                                    "name": pending["name"], "arguments": json.dumps(pending["args"])}}]}
                            finish = "tool_calls"
                    data = {"id": "chatcmpl-fsm", "created": int(time.time()), "model": "fsm",
                            "object": "chat.completion", "choices": [{"index": 0, "message": message,
                                                                        "finish_reason": finish}],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
                    if body.get("stream"):
                        delta = dict(message)
                        for i, call in enumerate(delta.get("tool_calls", [])):
                            call["index"] = i
                        data["object"] = "chat.completion.chunk"
                        data["choices"] = [{"index": 0, "delta": delta, "finish_reason": None}]
                        end = {**data, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                        payload = ("data: " + json.dumps(data) + "\n\ndata: " + json.dumps(end)
                                   + "\n\ndata: [DONE]\n\n").encode()
                        mime = "text/event-stream"
                    else:
                        payload = json.dumps(data).encode()
                        mime = "application/json"
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        try:
            self.temp = tempfile.TemporaryDirectory(prefix="fsm-opencode-")
            private = Path(self.temp.name)
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            provider_url = f"http://127.0.0.1:{self.server.server_port}/v1"
            config = {"$schema": "https://opencode.ai/config.json", "share": "disabled",
                      "enabled_providers": ["fsm-local"], "model": "fsm-local/fsm",
                      "small_model": "fsm-local/fsm", "plugin": [],
                      "provider": {"fsm-local": {"npm": "@ai-sdk/openai-compatible",
                          "options": {"baseURL": provider_url, "apiKey": self.secret},
                          "models": {"fsm": {"name": "fsm", "limit": {"context": 200000, "output": 4096}}}}},
                      "agent": {"fsm-executor": {"mode": "primary", "model": "fsm-local/fsm",
                          "prompt": "Execute the supplied tool call and stop.",
                          "tools": {"*": False, **{t: True for t in PRIMITIVES}},
                          "permission": {"*": "deny", **{t: "allow" for t in PRIMITIVES},
                                         "external_directory": {"*": "deny",
                                             str(self.workdir) + "/*": "allow",
                                             str(self.workdir.resolve()) + "/*": "allow"}}}}}
            env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
            env.update({"HOME": str(private), "XDG_CONFIG_HOME": str(private / "config"),
                        "XDG_DATA_HOME": str(private / "data"), "XDG_CACHE_HOME": str(private / "cache"),
                        "OPENCODE_SERVER_PASSWORD": self.secret,
                        "OPENCODE_CONFIG_CONTENT": json.dumps(config), "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
                        "OPENCODE_DISABLE_AUTOUPDATE": "true", "OPENCODE_DISABLE_LSP_DOWNLOAD": "true"})
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            self.log = open(private / "server.log", "w+")
            self.process = subprocess.Popen([self.binary, "serve", "--pure", "--hostname", "127.0.0.1",
                                             "--port", str(port)], cwd=self.workdir, env=env,
                                            stdout=self.log, stderr=self.log, start_new_session=True)
            self.client = httpx.Client(base_url=f"http://127.0.0.1:{port}",
                                       params={"directory": str(self.workdir)},
                                       timeout=self.timeout_s, trust_env=False, auth=("opencode", self.secret))
            deadline = time.monotonic() + min(self.timeout_s, 30)
            while True:
                try:
                    if self.client.get("/global/health", timeout=1).is_success:
                        break
                except httpx.TransportError:
                    pass
                if self.process.poll() is not None or time.monotonic() > deadline:
                    self.log.seek(0)
                    raise OpenCodeError("OpenCode startup failed: " + self.log.read()[-2000:])
                time.sleep(.1)
            self.available_tools = set(self._request("GET", "/experimental/tool/ids"))
            self.session = self._request("POST", "/session", {"title": "FSM native tools"})["id"]
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _request(self, method, path, body=None):
        r = self.client.request(method, path, json=body, timeout=self.timeout_s)
        if not r.is_success:
            raise OpenCodeError(f"OpenCode {path}: HTTP {r.status_code}: {r.text[:1000]}")
        return r.json()

    def describe_tools(self) -> dict[str, ToolSpec]:
        """Execution backend interface: the native tool definitions this OpenCode instance actually provides (the registry intersected with the installed tools)."""
        avail = getattr(self, "available_tools", None)
        return {n: s for n, s in REGISTRY.items() if avail is None or n in avail}

    def call(self, name: str, inp: dict) -> dict:
        if name not in PRIMITIVES:
            raise OpenCodeError(f"Unsupported native tool {name!r}; recompile legacy machines")
        with self.lock:
            if name not in getattr(self, "available_tools", ()):
                raise OpenCodeError(f"Installed OpenCode does not provide native tool {name!r}")
            if self.process is None or self.process.poll() is not None:
                raise OpenCodeError("OpenCode backend is not running")
            call_id = "call_" + uuid.uuid4().hex
            args = deepcopy(inp)
            self.pending = {"id": call_id, "name": name, "args": args}
            self.provider_error = ""
            try:
                reply = self._request("POST", f"/session/{self.session}/message", {
                    "agent": "fsm-executor", "model": {"providerID": "fsm-local", "modelID": "fsm"},
                    "parts": [{"type": "text", "text": f"Execute FSM call {call_id}."}]})
                if self.provider_error:
                    raise OpenCodeError(self.provider_error)
                # The final assistant message may follow the tool-bearing message.
                messages = self._request("GET", f"/session/{self.session}/message")
                parts = [p for msg in messages for p in msg.get("parts", [])
                         if p.get("type") == "tool" and p.get("callID") == call_id]
                if len(parts) != 1:
                    raise OpenCodeError(f"Expected one native tool result, got {len(parts)}: "
                                        + str(reply.get("info", {}).get("error", ""))[:800])
                out = tool_result(parts[0], name, args)
                self.calls.append({"name": name, "input": args, "output": out})
                return out
            except httpx.TimeoutException as exc:
                # A timed-out native action must not keep mutating files after the caller returns.
                try:
                    self.client.post(f"/session/{self.session}/abort", timeout=2)
                except httpx.HTTPError:
                    pass
                self.__exit__(None, None, None)
                raise OpenCodeError(f"OpenCode tool timed out after {self.timeout_s}s") from exc
            finally:
                self.pending = None

    def __exit__(self, *_args):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            except ProcessLookupError:
                pass
            self.process = None
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2)
            self.server = None
        if hasattr(self, "log"):
            self.log.close()
        if self.temp is not None:
            self.temp.cleanup()
            self.temp = None

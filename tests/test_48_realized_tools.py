"""Model-realized tools: a tool the machine has but the backend lacks is written by the model, from its registry definition, as one shell command that the backend executes."""
from __future__ import annotations

from hexis.tools.backends.realized import RealizedTools
from hexis.tools.toolspec import ToolSpec


class Native:
    available_tools = {"bash", "read"}

    def __init__(self):
        self.calls = []

    def call(self, name, inp):
        self.calls.append((name, inp))
        if name == "bash":
            return {"ok": True, "returncode": 0, "stdout": f"ran: {inp['command']}", "stderr": ""}
        return {"ok": True, "returncode": 0, "stdout": "read", "stderr": ""}


class Model:
    def __init__(self):
        self.prompts = []

    def generate(self, *, prompt, values, history=()):
        self.prompts.append(prompt)
        return {"command": f"python3 - <<'EOF'\nprint('inspect {values['arguments']['path']}')\nEOF"}


def test_abstract_tool_is_realized_through_the_model_and_shell():
    spec = ToolSpec(name="inspect_workbook", description="Inspect the workbook at `path`.",
                    input_schema={"path": {"type": "string", "required": True}}, source="registry",
                    output_schema={"ok": "boolean", "returncode": "integer", "stdout": "string"},
                    success="returncode == 0", primary="stdout")
    native, model = Native(), Model()
    tools = RealizedTools(native, {"inspect_workbook": spec}, model)
    assert tools.available_tools == {"bash", "read", "inspect_workbook"}
    out = tools.call("inspect_workbook", {"path": "/w/in.xlsx"})
    assert out["ok"] and "inspect /w/in.xlsx" in out["realized_command"]
    assert native.calls[0][0] == "bash" and "inspect /w/in.xlsx" in native.calls[0][1]["command"]
    assert "Inspect the workbook" in model.prompts[0] and "/w/in.xlsx" in model.prompts[0]
    assert tools.realized[0]["name"] == "inspect_workbook"
    # native tools are passed straight through, without calling the model
    tools.call("read", {"filePath": "/w/in.xlsx"})
    assert len(model.prompts) == 1 and native.calls[-1][0] == "read"


def test_realize_failure_is_a_failed_call_not_an_exception():
    class Broken(Model):
        def generate(self, **kw):
            raise RuntimeError("endpoint down")
    spec = ToolSpec(name="audit_workbook", source="registry", success="returncode == 0")
    out = RealizedTools(Native(), {"audit_workbook": spec}, Broken()).call("audit_workbook", {"x": 1})
    assert out["ok"] is False and out["returncode"] != 0 and "endpoint down" in out["stderr"]

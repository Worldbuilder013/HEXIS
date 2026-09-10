"""Build AWM workflows and ReasoningBank memories from trajectories.

    skill2fsm memory convert RUN_DIR TRAJ.jsonl       # skill-arm event streams of a bench run -> trajectories
    skill2fsm memory judge TRAJ.jsonl                 # model self-judgement of success, written back as self_judge
    skill2fsm memory awm TRAJ.jsonl WORKFLOWS.txt     # AWM offline induction from successful trajectories
    skill2fsm memory rbank TRAJ.jsonl BANK.jsonl      # ReasoningBank: strategies from successes, lessons from failures
    skill2fsm memory index BANK.jsonl                 # embedding index, written as BANK.npy
    skill2fsm memory precompute BANK.jsonl TASKS.yaml TASK_IDS OUT.json [--mode xlsx] [--k 5]
    skill2fsm memory retrieve BANK.jsonl "query" [--k 5]

Build memories from a development split and keep them frozen at test time. Success labels come from
the grader (``passed``); set the environment variable ``LABEL=self`` to use the model's
self-judgement instead. Model calls use the default endpoint profile (``MODEL`` / ``BASE_URL`` /
``API_KEY``). Embeddings need the ``memory`` extra (sentence-transformers).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _client():
    from skill2fsm.llm_client import client_from_env
    return client_from_env(timeout=600.0, max_retries=3)


def _ask(cl, prompt: str, max_tokens: int = 4096) -> str:
    return cl.complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens).text.strip()


# ------------------------------------------------------------------ convert
def convert(bench_dir: pathlib.Path, out: pathlib.Path) -> None:
    rows = {}
    for line in (bench_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("arm") == "skill":
                rows[r["task"]] = r
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for tid, r in sorted(rows.items()):
            d = bench_dir / "skill" / tid / f"r{r['rep']}"
            ev = d / "events.jsonl"
            if not ev.is_file():
                continue
            steps = []
            for raw in ev.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    e = json.loads(raw)
                except ValueError:
                    continue
                part = e.get("part") or {}
                t = part.get("type")
                if t == "text" and (part.get("text") or "").strip():
                    steps.append({"kind": "think", "text": part["text"].strip()[:1500]})
                elif t == "tool":
                    st = part.get("state") or {}
                    inp = st.get("input") or {}
                    cmd = inp.get("command") or inp.get("filePath") or json.dumps(inp, ensure_ascii=False)
                    steps.append({"kind": "action", "tool": part.get("tool"), "input": str(cmd)[:800],
                                  "output": str(st.get("output") or "")[:600]})
            prompt = (d / "prompt.txt").read_text(encoding="utf-8") if (d / "prompt.txt").is_file() else ""
            ans = (d / "answer.txt").read_text(encoding="utf-8").strip() if (d / "answer.txt").is_file() else ""
            fh.write(json.dumps({"task": tid, "prompt": prompt.strip(), "steps": steps, "answer": ans,
                                 "passed": bool(r.get("passed")), "why": r.get("why", "")}, ensure_ascii=False) + "\n")
            n += 1
    print(f"converted {n} trajectories -> {out}")


def _ok(r: dict) -> bool:
    """Success label: the grader's verdict by default (AWM's offline setting also uses labels);
    the model's self-judgement when LABEL=self."""
    return bool(r.get("self_judge")) if os.environ.get("LABEL") == "self" else bool(r["passed"])


def _render(traj: dict, max_chars: int = 9000) -> str:
    parts = [f"Task: {traj['prompt'][:1200]}"]
    for s in traj["steps"]:
        if s["kind"] == "think":
            parts.append(f"<think>\n{s['text']}\n</think>")
        else:
            parts.append(f"<action>\n{s['tool']}: {s['input']}\n</action>\n<observation>\n{s['output'][:300]}\n</observation>")
    if traj.get("answer"):
        parts.append(f"Final answer file: {traj['answer'][:300]}")
    text = "\n".join(parts)
    return text[:max_chars]


# ------------------------------------------------------------------ judge
JUDGE = """You are evaluating whether an agent completed a task correctly, using only the trajectory below (no ground truth is available). \
Consider whether the required output was produced, whether the steps are coherent, and whether the final result plausibly satisfies the request. \
Answer with exactly one word: SUCCESS or FAILURE.

{traj}
"""


def judge(traj_path: pathlib.Path) -> None:
    rows = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with _client() as cl:
        for r in rows:
            if "self_judge" in r:
                continue
            ans = _ask(cl, JUDGE.format(traj=_render(r)), max_tokens=16).upper()
            r["self_judge"] = "SUCCESS" in ans and "FAILURE" not in ans
            print(r["task"], "self:", r["self_judge"], "true:", r["passed"])
    traj_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    agree = sum(1 for r in rows if r["self_judge"] == r["passed"])
    print(f"self-judge agreement with grader: {agree}/{len(rows)}")


# ------------------------------------------------------------------ AWM
#: Offline workflow-induction instruction adapted from Agent Workflow Memory (Wang et al., 2024;
#: https://github.com/zorazrw/agent-workflow-memory, Apache-2.0). The task description is rewritten for
#: trajectories of shell and file tools; see THIRD_PARTY_NOTICES.md.
AWM_INSTRUCTION ="""Given a list of tasks solved by an agent with a shell and file tools, your task is to extract the common workflows to solve these tasks. \
Each given task contains a natural language instruction, and a series of reasoning steps and tool actions to solve the task. \
You need to find the repetitive subset of actions across multiple tasks, and extract each of them out as a workflow. \
Each workflow should be a commonly-reused sub-routine of the tasks. Do not generate similar or overlapping workflows. \
Each workflow should have at least two steps. Represent the non-fixed elements (file names, sheet names, values, question specifics) with descriptive variable names in curly braces. \
Keep the values of invariant elements (tool names, fixed commands, fixed output paths) as they will share and stay invariant across tasks. \
Try to generate as many workflows that can cover all the tasks in the input list.

Write each workflow as:
## Workflow: <short name>
<think>
<when to use it and the reasoning behind the steps>
</think>
<action>
<step 1 as a tool action with variables>
<step 2 ...>
</action>
"""


def _degenerate(text: str) -> bool:
    """A degenerate induction: no workflow heading at all, or one line repeated more than ten times in a row
    (small models looping on tags in long contexts)."""
    if "## Workflow" not in text:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    run = 1
    for x, y in zip(lines, lines[1:]):
        run = run + 1 if x == y else 1
        if run > 10:
            return True
    return False


def awm(traj_path: pathlib.Path, out: pathlib.Path, batch: int = 8) -> None:
    """AWM offline induction over all successful trajectories. When the single induction degenerates,
    induce per batch of ``batch`` trajectories and concatenate (AWM's implementation also induces per group)."""
    rows = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    good = [r for r in rows if _ok(r)]

    def induce(cl, group):
        examples = "\n\n".join(_render(r, 4000) for r in group)
        return _ask(cl, AWM_INSTRUCTION + "\n## Concrete Examples\n\n" + examples + "\n\n## Summary Workflows\n", max_tokens=6000)

    with _client() as cl:
        text = induce(cl, good)
        if _degenerate(text):
            print(f"AWM: induction over all {len(good)} trajectories degenerated; inducing in batches of {batch}")
            parts = []
            for i in range(0, len(good), batch):
                t = induce(cl, good[i:i + batch])
                if not _degenerate(t):
                    parts.append(t)
                else:
                    print(f"  batch {i // batch + 1} degenerated again; dropped")
            text = "\n\n".join(parts)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"AWM: induced from {len(good)}/{len(rows)} trajectories, {text.count('## Workflow')} workflows -> {out}")


# ------------------------------------------------------------------ ReasoningBank
RB_SUCCESS = """You are an expert at extracting transferable reasoning strategies from an agent's successful experience. \
Read the trajectory below and distill 1 to 3 memory items that would help the agent on similar future tasks. \
Each item must be general (not tied to this task's specific values), actionable, and grounded in what actually worked here.
Return a JSON list of objects with keys "title" (one line), "description" (one sentence), "content" (2 to 4 sentences of concrete guidance).

{traj}
"""
RB_FAILURE = """You are an expert at extracting lessons from an agent's failed experience. \
Read the trajectory below, identify why it failed (a wrong assumption, a skipped check, a misuse of a tool, a misread instruction), and distill 1 to 3 memory items \
that would help the agent avoid the same failure on similar future tasks. Each item must be general, actionable, and grounded in this trajectory.
Return a JSON list of objects with keys "title" (one line), "description" (one sentence), "content" (2 to 4 sentences of concrete guidance).

{traj}
"""


def _parse_items(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except ValueError:
        items = []
    return [i for i in items if isinstance(i, dict) and i.get("title") and i.get("content")]


def rbank(traj_path: pathlib.Path, out: pathlib.Path) -> None:
    rows = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = []
    with _client() as cl:
        for r in rows:
            ok = _ok(r)
            text = _ask(cl, (RB_SUCCESS if ok else RB_FAILURE).format(traj=_render(r)), max_tokens=2000)
            got = _parse_items(text)
            for it in got:
                it.update({"source_task": r["task"], "source_outcome": "success" if ok else "failure"})
            items.extend(got)
            print(r["task"], "success" if ok else "failure", len(got), "items")
    out.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
    print(f"ReasoningBank: {len(items)} items from {len(rows)} trajectories -> {out}")


def _embed(texts: list):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL)
    return np.asarray(m.encode(texts, normalize_embeddings=True))


def _load_bank(bank: pathlib.Path) -> list:
    return [json.loads(line) for line in bank.read_text(encoding="utf-8").splitlines() if line.strip()]


def index(bank: pathlib.Path) -> None:
    import numpy as np
    items = _load_bank(bank)
    vecs = _embed([f"{i['title']}. {i.get('description', '')}" for i in items])
    np.save(bank.with_suffix(".npy"), vecs)
    print(f"indexed {len(items)} items -> {bank.with_suffix('.npy')}")


def retrieve(bank: pathlib.Path, query: str, k: int = 5) -> list:
    import numpy as np
    items = _load_bank(bank)
    vecs = np.load(bank.with_suffix(".npy"))
    q = _embed([query])[0]
    order = np.argsort(-(vecs @ q))[:k]
    return [items[i] for i in order]


def format_memory(items: list) -> str:
    lines = ["## Relevant memory from past experience", ""]
    for i in items:
        lines.append(f"- **{i['title']}** — {i.get('description', '')}\n  {i['content']}")
    return "\n".join(lines) + "\n"


def precompute(bank: pathlib.Path, queries: dict, out: pathlib.Path, k: int = 5) -> None:
    """Retrieve for every test task ahead of time (task id -> injected text), so benchmark runs do not
    load the embedding model."""
    import numpy as np
    items = _load_bank(bank)
    vecs = np.load(bank.with_suffix(".npy"))
    tids = list(queries)
    qv = _embed([queries[t] for t in tids])
    res = {}
    for t, q in zip(tids, qv):
        order = np.argsort(-(vecs @ q))[:k]
        res[t] = format_memory([items[i] for i in order])
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"precomputed retrieval for {len(res)} tasks (k={k}) -> {out}")


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("convert", help="convert the skill-arm event streams of a bench run into trajectories")
    p.add_argument("run_dir", help="output directory of skill2fsm bench")
    p.add_argument("out", help="trajectory JSONL to write")
    p = sub.add_parser("judge", help="ask the model whether each trajectory succeeded (writes self_judge in place)")
    p.add_argument("traj", help="trajectory JSONL")
    p = sub.add_parser("awm", help="induce AWM workflows from successful trajectories")
    p.add_argument("traj", help="trajectory JSONL")
    p.add_argument("out", help="workflow text file to write")
    p = sub.add_parser("rbank", help="extract ReasoningBank memory items from all trajectories")
    p.add_argument("traj", help="trajectory JSONL")
    p.add_argument("out", help="memory bank JSONL to write")
    p = sub.add_parser("index", help="embed memory items (writes BANK.npy next to the bank)")
    p.add_argument("bank", help="memory bank JSONL")
    p = sub.add_parser("precompute", help="retrieve memories for each task and save the text to inject")
    p.add_argument("bank", help="memory bank JSONL (indexed)")
    p.add_argument("tasks_file", help="task YAML")
    p.add_argument("task_ids", help="file with comma- or newline-separated task ids")
    p.add_argument("out", help="retrieval JSON to write (task id -> text)")
    p.add_argument("--mode", choices=("xlsx", "livemath", "filetask"), default="livemath",
                   help="xlsx builds queries from the spreadsheet prompt; the other modes use the task text")
    p.add_argument("--k", type=int, default=5, help="memories per task")
    p = sub.add_parser("retrieve", help="print the titles of the top memories for a query")
    p.add_argument("bank", help="memory bank JSONL (indexed)")
    p.add_argument("query", help="query text")
    p.add_argument("--k", type=int, default=5, help="number of memories")
    a = ap.parse_args(argv)

    if a.cmd == "convert":
        convert(pathlib.Path(a.run_dir), pathlib.Path(a.out))
    elif a.cmd == "judge":
        judge(pathlib.Path(a.traj))
    elif a.cmd == "awm":
        awm(pathlib.Path(a.traj), pathlib.Path(a.out))
    elif a.cmd == "rbank":
        rbank(pathlib.Path(a.traj), pathlib.Path(a.out))
    elif a.cmd == "index":
        index(pathlib.Path(a.bank))
    elif a.cmd == "precompute":
        import yaml
        text = pathlib.Path(a.task_ids).read_text(encoding="utf-8")
        tids = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
        pool = {str(r["id"]): r for r in yaml.safe_load(pathlib.Path(a.tasks_file).read_text(encoding="utf-8"))}
        if a.mode == "xlsx":
            from skill2fsm.cli.collect import task_prompt
            queries = {t: task_prompt(pool[t]) for t in tids}
        else:
            queries = {t: str(pool[t]["turns"][0]) for t in tids}
        precompute(pathlib.Path(a.bank), queries, pathlib.Path(a.out), a.k)
    elif a.cmd == "retrieve":
        for it in retrieve(pathlib.Path(a.bank), a.query, a.k):
            print("-", it["title"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

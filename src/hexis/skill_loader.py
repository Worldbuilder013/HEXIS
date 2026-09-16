"""Read an Anthropic Agent Skill directory from disk and split the Markdown body into H2 sections.

An Agent Skill is a directory:

* ``SKILL.md`` -- YAML frontmatter (``name`` / ``description``, the public activation condition)
  followed by the Markdown body (the actual operating procedure). **What interpretive execution
  loads into the context is this body**, see :meth:`AgentSkill.full_text`.
* ``references/*.md`` -- supplementary documents (playbooks) loaded on demand.
* ``scripts/*.py`` -- executable scripts that ship with the skill.

This module is **independent of any particular skill**: it knows nothing about math or tables,
only the directory convention above. Which parts of the body count as "a clause" is up to the
skill itself.

Reference documents are stored here **as paths only, not contents**
(:attr:`AgentSkill.references`); read one with :meth:`AgentSkill.read_reference` when needed: a
skill's references can be large, and compilation usually reads only a few of them, so loading them
all into memory is pointless.

Line numbers are always **1-based, inclusive ranges**, and are **line numbers in the file** --
clauses must be traceable to "which line of which file", and an off-by-one breaks that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

SKILL_FILE = "SKILL.md"

#: frontmatter: the YAML block enclosed by ``---`` at the start of the file.
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---[ \t]*\n?(.*)$", re.DOTALL)

#: Start/end of a fenced code block (``` or ~~~). Skipped when splitting -- ``## `` inside code is not a heading.
_FENCE = re.compile(r"^\s*(```|~~~)")

_H1 = re.compile(r"^#[ \t]+(.+?)[ \t]*$")
_H2 = re.compile(r"^##[ \t]+(.+?)[ \t]*$")


# --------------------------------------------------------------------------- #
# Skill
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentSkill:
    """A parsed Agent Skill directory.

    ``description`` is the public surface (the activation condition the host always sees),
    ``body`` is the protected surface (the operating procedure the skill wants executed but not
    repeated back). This split does not affect the behaviour of this module, but it is a premise
    of the whole experiment, so the two are stored separately.

    ``slug`` is taken from **the frontmatter name** -- in the Agent Skill spec that field is itself
    a slug-shaped identifier (``solve-math-rigorously``) and is the skill's identity; the directory
    name is a product of vendoring and may differ (a vendored skill's directory often does), so use
    ``root.name`` if you want the directory name. Only when the frontmatter is missing does it fall
    back to the directory name. ``name`` is the human-readable title from the body's H1.
    """

    slug: str                                   # skill identifier: the frontmatter name
    name: str                                   # human-readable title: the body's H1
    description: str                            # the frontmatter description
    root: Path                                  # skill directory (absolute path)
    body: str                                   # SKILL.md body without the frontmatter
    frontmatter: dict = field(default_factory=dict)
    references: dict[str, Path] = field(default_factory=dict)   # by stem
    scripts: dict[str, Path] = field(default_factory=dict)      # by stem

    # ---- reading ---- #
    def read_reference(self, stem: str) -> str:
        """Read the full text of one ``references/*.md`` by stem. KeyError if there is no such document."""
        if stem not in self.references:
            raise KeyError(f"{self.slug} has no reference document {stem!r} (available: "
                           f"{sorted(self.references)})")
        return self.references[stem].read_text(encoding="utf-8")

    def full_text(self) -> str:
        """What interpretive execution loads into the context: **only the SKILL.md body**.

        Excludes the frontmatter (the public surface, which the host provides separately) and the
        references (loaded on demand; the model decides whether to read them).
        """
        return self.body

    def card_text(self) -> str:
        """The public card that always stays in the context: name + description."""
        return f"{self.name}: {self.description}"

    # ---- paths ---- #
    @property
    def skill_md(self) -> Path:
        """Path of SKILL.md. Clauses are traced back by **file line numbers**, so callers often need the raw text."""
        return self.root / SKILL_FILE

    def raw_skill_md(self) -> str:
        """The raw full text of SKILL.md (including frontmatter). Line numbers refer to this text."""
        return self.skill_md.read_text(encoding="utf-8")


def load_agent_skill(root: Path | str) -> AgentSkill:
    """Parse an Agent Skill directory. FileNotFoundError if SKILL.md is missing."""
    root = Path(root).resolve()          # make absolute: the caller's cwd should not affect parsing
    md = root / SKILL_FILE
    if not md.is_file():
        raise FileNotFoundError(f"no {SKILL_FILE} under {root}")

    raw = md.read_text(encoding="utf-8")
    m = _FRONTMATTER.match(raw)
    if m:
        fm = yaml.safe_load(m.group(1)) or {}
        body = m.group(2)
    else:
        fm, body = {}, raw
    if not isinstance(fm, dict):         # frontmatter that is not a mapping is treated as absent
        fm = {}

    slug = str(fm.get("name") or root.name)
    references = {p.stem: p for p in sorted((root / "references").glob("*.md"))} \
        if (root / "references").is_dir() else {}
    scripts = {p.stem: p for p in sorted((root / "scripts").glob("*.py"))} \
        if (root / "scripts").is_dir() else {}

    return AgentSkill(
        slug=slug,
        name=_h1(body) or slug,
        description=_one_line(str(fm.get("description") or "")),
        root=root,
        body=body,
        frontmatter=fm,
        references=references,
        scripts=scripts,
    )


# --------------------------------------------------------------------------- #
# Generic H2 splitting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Section:
    """One H2 section. ``line_start`` is the ``## `` line, ``line_end`` is the section's last line.

    ``index`` starts at 1, in document order. Content before the first H2 (the H1 title, the
    introduction) is **not** a section; callers that need it take it themselves via
    ``sections(md)[0].line_start``.
    """

    index: int
    title: str
    line_start: int
    line_end: int
    text: str


def sections(md: str) -> list[Section]:
    """Split Markdown at ``## ``. A ``## `` inside a fenced code block is not a heading.

    Line numbers are 1-based within **the text passed in**: pass the raw text to get raw-text line
    numbers, pass the body to get body line numbers. Callers that want file line numbers pass the
    raw text.
    """
    lines = md.splitlines()
    heads: list[tuple[int, str]] = []          # (1-based line number, title)
    for i, line in enumerate(mask_fences(lines), start=1):
        if line is None:                       # inside a fence
            continue
        m = _H2.match(line)
        if m:
            heads.append((i, m.group(1).strip()))

    out: list[Section] = []
    for k, (start, title) in enumerate(heads):
        end = heads[k + 1][0] - 1 if k + 1 < len(heads) else len(lines)
        end = trim_blank_tail(lines, start, end)
        out.append(Section(index=k + 1, title=title, line_start=start,
                           line_end=end, text="\n".join(lines[start - 1:end])))
    return out


def mask_fences(lines: list[str]) -> list[str | None]:
    """Replace lines inside fenced code blocks with ``None``, leaving the rest as is. Heading lookup and splitting all go through this layer.

    A general utility: any code that "looks for Markdown structure line by line" must mask code
    blocks first, otherwise a ``## `` in an example command is taken for a heading. Clause
    splitting reuses it.
    """
    out: list[str | None] = []
    fence: str | None = None
    for line in lines:
        m = _FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)
            out.append(None)
            continue
        if fence is not None:
            out.append(None)
            if m and m.group(1) == fence:
                fence = None
            continue
        out.append(line)
    return out


def trim_blank_tail(lines: list[str], start: int, end: int) -> int:
    """Trim blank lines off the end of ``[start, end]`` (always keeping the ``start`` line). Line numbers are 1-based."""
    while end > start and not lines[end - 1].strip():
        end -= 1
    return end


def _h1(body: str) -> str:
    """The first ``# `` heading in the body (the human-readable title). Empty string if none. Headings inside fences don't count."""
    for line in mask_fences(body.splitlines()):
        if line is not None and (m := _H1.match(line)):
            return m.group(1).strip()
    return ""


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()



def markdown_clauses(doc: str) -> list[tuple[str, str, str]]:
    """A clause splitter that works on **any** SKILL.md: ``(id, original text, location)``.

    Numbering follows the structure: H1 is the title and gets no number; H2s are S1, S2, ... in
    order; H3s below them are S2.1, S2.2, ...; each list item under a heading goes one level
    further, S2.1.1, ... This matches the position-based numbering used in hand-written reference
    machine ledgers. A document without headings is S1 as a whole.
    The location is ``doc:<line number>`` (relative to the text passed in); it does not pretend to
    be a file line number.
    """
    lines = (doc or "").splitlines()
    out: list[tuple[str, str, str]] = []
    h2 = h3 = item = 0
    cur_head: Optional[str] = None
    buf: list[str] = []
    buf_line = 0

    def flush() -> None:
        # Body text that appears under the same heading after a list is merged back into that
        # heading's entry: ids must be unique, and there cannot be two S2 clauses.
        nonlocal buf
        if cur_head and buf:
            text = "\n".join(buf).strip()
            for k, (cid, old, loc) in enumerate(out):
                if cid == cur_head:
                    out[k] = (cid, old + "\n" + text, loc)
                    break
            else:
                out.append((cur_head, text, f"doc:{buf_line}"))
        buf = []

    for i, ln in enumerate(lines, 1):
        st = ln.strip()
        if st.startswith("## ") and not st.startswith("### "):
            flush(); h2 += 1; h3 = 0; item = 0
            cur_head, buf, buf_line = f"S{h2}", [st], i
        elif st.startswith("### "):
            flush(); h3 += 1; item = 0
            cur_head, buf, buf_line = f"S{max(h2, 1)}.{h3}", [st], i
        elif st[:2] in ("- ", "* ") or (st[:3].rstrip(".").isdigit() and ". " in st[:4]):
            flush(); item += 1
            base = f"S{max(h2, 1)}" + (f".{h3}" if h3 else "")
            out.append((f"{base}.{item}", st, f"doc:{i}"))
            buf, buf_line = [], 0
            if cur_head is None:
                cur_head = base
        elif st.startswith("# "):
            flush(); cur_head = None
        elif st:
            if cur_head is None:
                cur_head, buf_line = "S1", i
            buf.append(st)
    flush()
    return out


__all__ = ["markdown_clauses", "AgentSkill", "SKILL_FILE", "Section", "load_agent_skill",
           "mask_fences", "sections", "trim_blank_tail"]

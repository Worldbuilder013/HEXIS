"""从磁盘读一个 Anthropic Agent Skill 目录，并把 Markdown 正文按 H2 切段。

一个 Agent Skill 就是一个目录：

* ``SKILL.md`` —— YAML frontmatter（``name`` / ``description``，公开的激活条件）后跟
  Markdown 正文（真正的操作规程）。**解释执行模式装进上下文的就是这份正文**，见
  :meth:`AgentSkill.full_text`。
* ``references/*.md`` —— 按需加载的补充文档（playbook）。
* ``scripts/*.py`` —— 技能自带的可执行脚本。

这个模块**与具体技能无关**：它不认识数学、不认识表格，只认识上面这套目录约定。哪一段
正文算「一条条款」是技能自己的事——见 :mod:`skill2fsm.examples.math_skill.clauses`。

参考文档正文这里**只存路径不存内容**（:attr:`AgentSkill.references`），要用再
:meth:`AgentSkill.read_reference`：一个技能的 references 可以很大，编译期真正读到的往往
只有几份，全量读进内存没有意义。

行号一律 **1 起、闭区间**，且是**文件里的行号**——条款要能溯源到「哪个文件第几行」，
偏移一位就断了。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SKILL_FILE = "SKILL.md"

#: frontmatter：文件开头的 ``---`` 围起来的 YAML 块。
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---[ \t]*\n?(.*)$", re.DOTALL)

#: 围栏代码块的起止（``` 或 ~~~）。切段时要跳过——代码里的 ``## `` 不是标题。
_FENCE = re.compile(r"^\s*(```|~~~)")

_H1 = re.compile(r"^#[ \t]+(.+?)[ \t]*$")
_H2 = re.compile(r"^##[ \t]+(.+?)[ \t]*$")


# --------------------------------------------------------------------------- #
# 技能
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentSkill:
    """一个已解析的 Agent Skill 目录。

    ``description`` 是公开面（宿主始终看得见的激活条件），``body`` 是受保护面（技能希望
    被执行、但不希望被复述的操作规程）。这个划分不影响本模块的行为，但它是整套实验的
    前提，所以两者分开存。

    ``slug`` 取 **frontmatter 的 name**——Agent Skill 规范里那个字段本身就是 slug 形态的
    标识（``solve-math-rigorously``），它才是技能的身份；目录名是 vendoring 的产物，两者
    可以不一样（本仓库的 ``third_party/math-skill/`` 就不一样），要目录名用
    ``root.name``。缺 frontmatter 时才退回目录名。``name`` 取正文 H1 的人读标题。
    """

    slug: str                                   # 技能标识：frontmatter 的 name
    name: str                                   # 人读标题：正文的 H1
    description: str                            # frontmatter 的 description
    root: Path                                  # 技能目录（绝对路径）
    body: str                                   # SKILL.md 去掉 frontmatter 的正文
    frontmatter: dict = field(default_factory=dict)
    references: dict[str, Path] = field(default_factory=dict)   # 按 stem
    scripts: dict[str, Path] = field(default_factory=dict)      # 按 stem

    # ---- 读取 ---- #
    def read_reference(self, stem: str) -> str:
        """按 stem 读一份 ``references/*.md`` 的全文。没有这份就 KeyError。"""
        if stem not in self.references:
            raise KeyError(f"{self.slug} 没有参考文档 {stem!r}（有 "
                           f"{sorted(self.references)}）")
        return self.references[stem].read_text(encoding="utf-8")

    def full_text(self) -> str:
        """解释执行模式装进上下文的东西：**只有 SKILL.md 正文**。

        不含 frontmatter（那是公开面，宿主另外给），也不含 references（那是按需加载的，
        模型自己决定要不要读）。
        """
        return self.body

    def card_text(self) -> str:
        """始终留在上下文里的公开卡片：name + description。"""
        return f"{self.name}: {self.description}"

    # ---- 路径 ---- #
    @property
    def skill_md(self) -> Path:
        """SKILL.md 的路径。条款要按**文件行号**溯源，所以调用方常要原文。"""
        return self.root / SKILL_FILE

    def raw_skill_md(self) -> str:
        """SKILL.md 的原始全文（含 frontmatter）。行号以它为准。"""
        return self.skill_md.read_text(encoding="utf-8")


def load_agent_skill(root: Path | str) -> AgentSkill:
    """解析一个 Agent Skill 目录。缺 SKILL.md 即 FileNotFoundError。"""
    root = Path(root).resolve()          # 绝对化：调用方的 cwd 不该影响解析结果
    md = root / SKILL_FILE
    if not md.is_file():
        raise FileNotFoundError(f"{root} 下没有 {SKILL_FILE}")

    raw = md.read_text(encoding="utf-8")
    m = _FRONTMATTER.match(raw)
    if m:
        fm = yaml.safe_load(m.group(1)) or {}
        body = m.group(2)
    else:
        fm, body = {}, raw
    if not isinstance(fm, dict):         # frontmatter 不是映射就当没有
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
# 通用 H2 切段
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Section:
    """一段 H2 小节。``line_start`` 是 ``## `` 那一行，``line_end`` 是本段最后一行。

    ``index`` 从 1 起，按文档顺序。第一个 H2 之前的内容（H1 标题、导言）**不是**小节，
    要它的调用方自己用 ``sections(md)[0].line_start`` 去取。
    """

    index: int
    title: str
    line_start: int
    line_end: int
    text: str


def sections(md: str) -> list[Section]:
    """把 Markdown 按 ``## `` 切段。围栏代码块里的 ``## `` 不算标题。

    行号是**传进来的这份文本**里的 1 起行号：传原文得原文行号，传正文得正文行号。
    调用方要文件行号就把原文传进来。
    """
    lines = md.splitlines()
    heads: list[tuple[int, str]] = []          # (行号 1 起, 标题)
    for i, line in enumerate(mask_fences(lines), start=1):
        if line is None:                       # 围栏内
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
    """把围栏代码块里的行换成 ``None``，其余原样。找标题/切段统一走这一层。

    通用工具：任何「按行找 Markdown 结构」的代码都得先把代码块屏蔽掉，否则示例命令里的
    ``## `` 会被当成标题。条款切分（examples/math_skill/clauses.py）复用它。
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
    """把 ``[start, end]`` 末尾的空行剪掉（至少保留 ``start`` 那一行）。行号 1 起。"""
    while end > start and not lines[end - 1].strip():
        end -= 1
    return end


def _h1(body: str) -> str:
    """正文里第一个 ``# `` 标题（人读标题）。没有就空串。围栏里的不算。"""
    for line in mask_fences(body.splitlines()):
        if line is not None and (m := _H1.match(line)):
            return m.group(1).strip()
    return ""


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


__all__ = ["AgentSkill", "SKILL_FILE", "Section", "load_agent_skill",
           "mask_fences", "sections", "trim_blank_tail"]

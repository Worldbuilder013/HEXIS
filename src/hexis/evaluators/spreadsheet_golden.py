# SPDX-License-Identifier: MIT AND CC-BY-SA-4.0
# The value normalization, cell comparison and range helpers and the LibreOffice recalculation command
# are adapted from SpreadsheetBench (https://github.com/RUCKBReasoning/SpreadsheetBench, CC BY-SA 4.0);
# see THIRD_PARTY_NOTICES.md.
"""SpreadsheetBench 的金标准判定。

判定是：把智能体保存下来的工作簿与金标准工作簿，在该任务声明的 ``answer_position``
区域内**逐格比对取值**。写错一格就是不通过。

比对逻辑逐字取自 SpreadsheetBench 官方的 ``evaluation.py``（见
``third_party/SpreadsheetBench/evaluation.py``），包括它那套类型归一：数值四舍五入到两位、
日期折算成 Excel 序列号、``""`` 与 ``None`` 视为相同。不自己另写一套——口径一旦漂移，
报出来的数就不再是 SpreadsheetBench 的那个数。

**通过率不能拿绝对值当闸门。** SpreadsheetBench 上最好的模型也只有五成上下，对编译产物
要求 90% 的绝对通过率是不可能达到的。所以闸门问的是**相对**问题：同一个智能体、同一个模型、同一个验证器下，编译产物通过的任务集合能不能
跟上原技能。见 :func:`relative_success`。
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

#: 官方数据集的位置。只读，不进 git（见 .gitignore）。
BENCH_ROOT = Path("third_party/SpreadsheetBench/spreadsheetbench_verified_400")


class GoldenUnavailable(RuntimeError):
    """拿不到金标准数据。**不降级**——降级判出来的数不再是 SpreadsheetBench 的口径。"""


# --------------------------------------------------------------------------- #
# 取值比对：逐字取自 SpreadsheetBench 的 evaluation.py（CC BY-SA 4.0，见 THIRD_PARTY_NOTICES.md）
# --------------------------------------------------------------------------- #
def _datetime_to_float(dt: datetime.datetime) -> float:
    excel_start_date = datetime.datetime(1899, 12, 30)
    delta = dt - excel_start_date
    return delta.days + delta.seconds / 86400.0


def _transform_value(v: Any) -> Any:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = round(float(v), 2)
    elif isinstance(v, datetime.time):
        v = str(v)[:-3]
    elif isinstance(v, datetime.datetime):
        v = round(_datetime_to_float(v), 0)
    elif isinstance(v, str):
        try:
            v = round(float(v), 2)
        except ValueError:
            pass
    return v


def compare_cell_value(v1: Any, v2: Any) -> bool:
    """两个单元格取值是否相同。空串与 ``None`` 视为相同，这是官方的判法。"""
    v1, v2 = _transform_value(v1), _transform_value(v2)
    if (v1 == "" and v2 is None) or (v1 is None and v2 == ""):
        return True
    if (v1 == "" and v2 == "") or (v1 is None and v2 is None):
        return True
    if type(v1) is not type(v2):
        return False
    return v1 == v2


def _col_num2name(n: int) -> str:
    name = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        name = chr(65 + rem) + name
    return name


def _col_name2num(name: str) -> int:
    num = 0
    for c in name:
        num = num * 26 + (ord(c) - ord("A") + 1)
    return num


def _parse_cell_range(range_str: str) -> tuple[tuple[int, int], tuple[int, int]]:
    start_cell, end_cell = range_str.split(":")
    sc = "".join(ch for ch in start_cell if not ch.isdigit())
    sr = "".join(ch for ch in start_cell if ch.isdigit())
    ec = "".join(ch for ch in end_cell if not ch.isdigit())
    er = "".join(ch for ch in end_cell if ch.isdigit())
    return (_col_name2num(sc), int(sr)), (_col_name2num(ec), int(er))


def cell_names(range_str: str) -> list[str]:
    """``'G2:G16'`` → ``['G2', ..., 'G16']``。单格直接返回它自己。"""
    if ":" not in range_str:
        return [range_str]
    (sc, sr), (ec, er) = _parse_cell_range(range_str)
    cols = [_col_num2name(i) for i in range(sc, ec + 1)]
    return [f"{c}{r}" for c in cols for r in range(sr, er + 1)]


# --------------------------------------------------------------------------- #
# 重算：openpyxl 不算公式，不重算就只能读到 None
# --------------------------------------------------------------------------- #
#: LibreOffice 的可执行文件。SpreadsheetBench 官方评测的前置依赖，
#: ``brew install --cask libreoffice`` / ``apt install libreoffice-calc``。
_SOFFICE_CANDIDATES = (
    "libreoffice", "soffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/usr/bin/soffice", "/usr/bin/libreoffice",
)


def find_soffice() -> Optional[str]:
    import shutil as _sh

    for c in _SOFFICE_CANDIDATES:
        hit = _sh.which(c)
        if hit:
            return hit
        if Path(c).is_file():
            return c
    return None


def recalculate(path: Path, *, timeout_s: float = 180.0) -> tuple[bool, str]:
    """用 LibreOffice 无头打开再存回，把公式的计算结果落成缓存值。

    **这一步不是可选的。** openpyxl 只写公式串、从不计算，``data_only=True`` 读回来就是
    ``None``。少了它，一份完全正确的答案会被判成失败——实测基线臂上 ``sb_49801`` 写出
    ``=LEFT(A1,2)&RIGHT(A1,5)``（正确），判定却报「金标准 'SL-0035'，产出 None」，八条任务
    的通过率因此从真实水平掉到 2/8。失败的形状是清一色的「产出 None」而不是「算错了」，
    那正是这个 bug 的指纹：``None`` 不是一个错误答案，是**没有值**。

    做法与调用参数逐字取自官方的 ``evaluation/open_spreadsheet.py``
    （``--headless --calc --convert-to xlsx:Calc MS Excel 2007 XML``），不自己另发明一套。

    返回 ``(成功与否, 原因)``。**失败不静默**：调用方必须把它变成一条明说的判定失败，
    而不是退回去比未重算的值——那等于把这个 bug 重新引入一次。
    """
    import shutil as _sh
    import subprocess
    import tempfile as _tf

    soffice = find_soffice()
    if soffice is None:
        return False, ("找不到 LibreOffice，无法重算公式。"
                       "装它：brew install --cask libreoffice（macOS）/ "
                       "apt install libreoffice-calc（Linux）。"
                       "不重算的话写公式的答案一律被读成 None，判定会系统性偏低。")
    path = Path(path).resolve()
    if not path.is_file():
        return False, f"文件不存在：{path}"
    with _tf.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [soffice, "--headless", "--calc",
                 "--convert-to", "xlsx:Calc MS Excel 2007 XML",
                 "--outdir", tmp, str(path)],
                capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"重算超时（{timeout_s:.0f}s）"
        out = Path(tmp) / (path.stem + ".xlsx")
        if proc.returncode != 0 or not out.is_file():
            return False, f"LibreOffice 退出码 {proc.returncode}：{(proc.stderr or '')[-200:]}"
        _sh.move(str(out), str(path.with_suffix(".xlsx")))
    return True, ""


def compare_workbooks(golden: Path, produced: Path, answer_position: str,
                      *, recalc: bool = True) -> tuple[bool, str]:
    """在 ``answer_position`` 区域内逐格比对取值。

    ``answer_position`` 形如 ``G2:G16`` 或 ``'Sheet2'!A1:B9``，逗号分隔多段，全部命中才算过。
    ``data_only=True`` 读的是**缓存值**而不是公式串：任务问的是结果对不对，用什么公式算出来
    的不在判定范围内。
    """
    import openpyxl

    if not Path(produced).is_file():
        return False, "没有产出工作簿"
    if recalc:
        ok, why = recalculate(Path(produced))
        if not ok:
            # 重算不了就判不通过并**说明白原因**，不退回去比未重算的值。
            return False, f"无法重算公式：{why}"
        produced = Path(produced).with_suffix(".xlsx")
    try:
        wb_gt = openpyxl.load_workbook(filename=str(golden), data_only=True)
        wb_pr = openpyxl.load_workbook(filename=str(produced), data_only=True)
    except Exception as exc:  # noqa: BLE001 — 打不开就是没通过，不该炸掉整批
        return False, f"{type(exc).__name__}: {exc}"

    for piece in answer_position.split(","):
        piece = piece.strip()
        if "!" in piece:
            sheet_name, cell_range = piece.split("!")
        else:
            sheet_name, cell_range = wb_gt.sheetnames[0], piece
        sheet_name = sheet_name.strip().strip("'")
        cell_range = cell_range.strip().strip("'")
        if sheet_name not in wb_pr.sheetnames:
            return False, f"产出里没有工作表 {sheet_name!r}"
        ws_gt, ws_pr = wb_gt[sheet_name], wb_pr[sheet_name]
        for name in cell_names(cell_range):
            a, b = ws_gt[name].value, ws_pr[name].value
            if not compare_cell_value(a, b):
                return False, f"{sheet_name}!{name} 金标准 {a!r}，产出 {b!r}"
    return True, ""


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
@dataclass
class GoldenTask:
    """一条 SpreadsheetBench 任务：指令、初始工作簿、金标准、判定区域。"""

    id: str
    instruction: str
    answer_position: str
    instruction_type: str
    init: Path
    golden: Path

    def verify(self, produced: Path) -> tuple[bool, str]:
        return compare_workbooks(self.golden, produced, self.answer_position)


def load_dataset(root: Path = BENCH_ROOT) -> dict[str, GoldenTask]:
    """按任务号索引整个 verified-400。

    每个任务目录里有三份测试用例（``1_``/``2_``/``3_`` 前缀），官方评测三份都要过。这里
    只取第一份：闭环要跑的是编译产物在**同一条任务**上跟不跟得上原技能，三份用例是同一
    条指令的三份数据，多跑两份只是把 token 乘三，判别力不增加。这一点在报出的数里要写明。
    """
    root = Path(root)
    meta = root / "dataset.json"
    if not meta.is_file():
        raise GoldenUnavailable(
            f"找不到 {meta}。SpreadsheetBench 的数据不随仓库分发，"
            "从 https://github.com/RUCKBReasoning/SpreadsheetBench 取 "
            "data/spreadsheetbench_verified_400.tar.gz 解压到 third_party/SpreadsheetBench/。")
    out: dict[str, GoldenTask] = {}
    for row in json.loads(meta.read_text("utf-8")):
        tid = str(row["id"])
        d = root / "spreadsheet" / tid
        init, golden = d / f"1_{tid}_init.xlsx", d / f"1_{tid}_golden.xlsx"
        if not (init.is_file() and golden.is_file()):
            continue
        out[tid] = GoldenTask(
            id=tid, instruction=row["instruction"],
            answer_position=row["answer_position"],
            instruction_type=row.get("instruction_type", ""),
            init=init, golden=golden)
    if not out:
        raise GoldenUnavailable(f"{root} 里一条完整任务都没有")
    return out


# --------------------------------------------------------------------------- #
# 相对通过率
# --------------------------------------------------------------------------- #
@dataclass
class SuccessReport:
    """一次判定的结果，逐条留痕。"""

    per_task: dict[str, dict] = field(default_factory=dict)
    oracle_passed: list[str] = field(default_factory=list)
    candidate_passed: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.per_task)

    @property
    def oracle_rate(self) -> Optional[float]:
        return len(self.oracle_passed) / self.n if self.n else None

    @property
    def candidate_rate(self) -> Optional[float]:
        return len(self.candidate_passed) / self.n if self.n else None

    def to_dict(self) -> dict:
        return {
            "metric": "SpreadsheetBench golden-workbook success",
            "n": self.n,
            "oracle_passed": sorted(self.oracle_passed),
            "candidate_passed": sorted(self.candidate_passed),
            "oracle_success_rate": self.oracle_rate,
            "candidate_success_rate": self.candidate_rate,
            "relative_success": relative_success(self),
            "regressions": sorted(set(self.oracle_passed) - set(self.candidate_passed)),
            "tasks": self.per_task,
        }


def relative_success(report: SuccessReport) -> Optional[float]:
    """编译产物在**原技能做得成的那些任务**上的通过比例。

    分母是原技能通过的任务数，不是全部任务数。原技能本来就做不成的任务，编译产物做不成
    不能算工具化的账——那是这条基准本身的难度，SpreadsheetBench 上最好的模型也只有五成
    上下。闸门要问的是「搬进代码之后，原本干得成的还干不干得成」。

    原技能一条都没通过时返回 ``None``（测不出），而不是 1.0。「没有回归」和「根本没测出
    东西」在数字上必须分得开——上一版正是在这种地方把跑失败读成了防护完美。
    """
    if not report.oracle_passed:
        return None
    kept = set(report.oracle_passed) & set(report.candidate_passed)
    return len(kept) / len(report.oracle_passed)


def find_produced(root: Path, *, expected: str = "", shipped: Iterable[Path] = ()) -> Optional[Path]:
    """从导出的产物里挑出智能体保存的那个工作簿。

    **按名字挑，不按时间挑。** 第一版按「最后修改的 .xlsx」挑，当场挑错：技能的
    ``assets/`` 里躺着全部八条任务的初始工作簿，沙箱开局把整个技能目录复制一遍，八份的
    mtime 因此都是新的，「最后修改」在它们之间基本等于随机。实测 sb_53449 就挑中了别的
    任务那份没动过的初始文件，判定报出 ``G4 金标准 None，产出 'd'``——那不是智能体做错了，
    是判定拿错了文件比。

    所以顺序是：任务里点名的那个输出名 → 排除掉与随技能分发的初始工作簿**字节相同**的
    文件之后剩下的最新一个 → 都没有就返回 ``None``，由调用方判为不通过。
    """
    root = Path(root)
    if not root.is_dir():
        return None
    xlsx = [f for f in root.rglob("*.xlsx") if not f.name.startswith("~$")]
    if not xlsx:
        return None
    if expected:
        hit = [f for f in xlsx if f.name == expected]
        if hit:
            return hit[0]
    # 随技能分发的那些初始工作簿逐字节剔掉。就地改写过的那一份字节已经变了，会留下来。
    seen = {f.read_bytes() for f in shipped if Path(f).is_file()}
    fresh = [f for f in xlsx if f.read_bytes() not in seen]
    if not fresh:
        return None
    return max(fresh, key=lambda f: f.stat().st_mtime)

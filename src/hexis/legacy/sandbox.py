"""跑真代码的沙箱：技能自带的 ``scripts/*.py`` 与模型现写的 python 都在这里执行。

编译一个技能绕不开**真的把它的脚本跑起来**。math-skill 的 ``scripts/math_verify.py``
是 788 行 sympy，里面 ``parse_expr`` 一路走到 python 的 ``eval``，而喂进去的表达式来自
大模型。这一层是整条链上**唯一**的边界，所以它按边界写，不按工具写。

四条真的落到实处的隔离，外加一条没落实、必须照实说的：

* **墙钟超时 + 杀进程树。** ``equiv "9^9^9" "1"`` 在这台机器上永不返回——那份脚本从头
  到尾没有任何内部超时。只 kill 父进程不够：模型现写的代码会派生子进程。win32 走
  ``taskkill /F /T`` 再补一刀 ``TerminateJobObject``，POSIX 走进程组 ``killpg``；杀完
  回头确认真的没了（:func:`pid_alive`），确认不了不谎报。
* **环境饥饿。** 子进程的环境**从空字典搭起**，只放白名单里那几个。``API_KEY`` /
  ``ANTHROPIC_*`` / ``OPENAI_*`` / ``HTTP(S)_PROXY`` 连调用方在 ``env_allow`` 里显式点名
  也不放行（:func:`_denied`）——一次手滑就是一把线上密钥进了 LLM 现写的 python。
* **内存上限。** POSIX 上 ``setrlimit(RLIMIT_AS)``；win32 上 Job Object
  （``CreateJobObjectW`` + ``ProcessMemoryLimit`` + ``KILL_ON_JOB_CLOSE``）。Job 建不起来
  或者进程挂不进去，就**降级并如实上报**：:meth:`Sandbox.isolation_report` 里
  ``mem_limit`` 变 ``False``、``mem_limit_method`` 变 ``"none"``。不声称没做到的事。
* **cwd 隔离。** 每个沙箱一个临时工作目录，绝不是仓库；脚本按 ``root`` 解析并校验不许用
  ``..`` 逃出去（逃逸直接 :class:`SandboxError`，不是安静地返回失败）。
  ``PYTHONDONTWRITEBYTECODE=1`` 顺带保证 ``third_party/`` 不会被写进 ``__pycache__``
  ——那棵树是编译目标，必须逐字节不变。

* **网络：拦不住。** 不上容器或网络命名空间，就没有办法在 OS 层给一个子进程断网，这里
  不装作能。``isolation_report()["network_blocked"]`` 恒为 ``False``，让 experiment.py
  把**真实的**隔离等级印进报告，而不是印一个好看的。同理 ``filesystem_isolated`` 也是
  ``False``：只换了 cwd，子进程照样能读整块盘。

调用约定上有两处是踩出来的，写在这里免得再踩：

* ``math_verify.py`` 的 ``--json`` 挂在**顶层** parser 上，必须排在子命令**前面**。
  ``--json equiv a b`` 退出码 0；``equiv a b --json`` 退出码 2（unrecognized arguments）。
* 不给 ``PYTHONIOENCODING=utf-8``，Windows 上 python 打印非 ASCII 直接
  ``UnicodeEncodeError``，从外面看只是「输出为空」——一个查不动的假故障。

**一次运行一个沙箱**（沿用旧 toolize.runner 的政策）：第三步要读第一步落下的中间文件，
每步各起一个沙箱就读不到了。同理这一份**不是线程安全的**：win32 的杀树会连坐同一个
Job 里的全部进程，一个沙箱同时跑两个命令会互相误杀。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from hexis.traces.trace_adapter import ERROR_RC, TIMEOUT_RC

OUTCOME_OK = "ok"
OUTCOME_NONZERO = "nonzero"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_ERROR = "error"

#: 从父进程环境里**按名字**放行的那几个。别的一律不进子进程。
_BASE_ALLOW = (
    "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "SYSTEMDRIVE",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "LANG", "LC_ALL", "TZ",
)

#: 无论调用方怎么点名都不放行的。名字里带这些子串的一律扣下。
_DENY_SUBSTR = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL")
_DENY_PREFIX = ("ANTHROPIC_", "OPENAI_", "AWS_", "AZURE_", "GOOGLE_", "GCP_",
                "HF_")
_DENY_EXACT = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY", "NO_PROXY")


class SandboxError(RuntimeError):
    """沙箱拒绝执行：路径逃逸、沙箱已关闭这类**调用方的错**。

    与「脚本跑挂了」刻意分开：后者是 :class:`ExecResult` 里 ``outcome='error'`` 的一条
    数据，前者是异常。理由沿用旧 toolize.runner 的那条政策——守卫抛出来的东西当成
    ``False`` 处理，机器就会安静地走错边，最后只报一句「结果不对」。
    """


def _denied(name: str) -> bool:
    """这个环境变量名是否属于「点名也不给」的那一类。"""
    up = name.upper()
    return (up in _DENY_EXACT
            or any(up.startswith(p) for p in _DENY_PREFIX)
            or any(s in up for s in _DENY_SUBSTR))


# --------------------------------------------------------------------------- #
# win32：Job Object（内存上限 + 关闭即连坐杀光）
# --------------------------------------------------------------------------- #
_JOB_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_EXTENDED_LIMIT_CLASS = 9
_STILL_ACTIVE = 259

if sys.platform == "win32":                       # pragma: no cover - 平台分支
    import ctypes
    from ctypes import wintypes

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong)]

    class _JOB_BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),      # ULONG_PTR
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _JOB_EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _JOB_BASIC_LIMIT),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    def _kernel32() -> Any:
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        # restype 必须显式给：默认 c_int 会把 64 位句柄截成 32 位，句柄看着有效、用起来炸。
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              wintypes.LPVOID, wintypes.DWORD]
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        return k


def _win_create_job(mem_limit_mb: int) -> tuple[Optional[int], str]:
    """建一个带单进程内存上限、句柄一关就连坐杀光的 Job。失败返回 ``(None, 原因)``。"""
    if sys.platform != "win32":
        return None, "非 win32"
    try:                                          # pragma: no cover - 平台分支
        k = _kernel32()
        handle = k.CreateJobObjectW(None, None)
        if not handle:
            return None, f"CreateJobObjectW 失败 err={ctypes.get_last_error()}"
        info = _JOB_EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_LIMIT_PROCESS_MEMORY | _JOB_LIMIT_KILL_ON_JOB_CLOSE)
        info.ProcessMemoryLimit = max(1, int(mem_limit_mb)) * 1024 * 1024
        ok = k.SetInformationJobObject(handle, _JOB_EXTENDED_LIMIT_CLASS,
                                       ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            err = ctypes.get_last_error()
            k.CloseHandle(handle)
            return None, f"SetInformationJobObject 失败 err={err}"
        return int(handle), ""
    except Exception as exc:                      # noqa: BLE001 - 降级，不是崩
        return None, f"{type(exc).__name__}: {exc}"


def _win_assign_job(job: int, pid: int) -> str:
    """把已经起来的进程挂进 Job。返回空串表示成功，否则是失败原因。

    这里有一个**真实存在的竞态**：进程是先起来、后挂进 Job 的，中间那几十毫秒（python
    解释器启动）里它派生的孙进程不受 Job 管。要彻底堵上得用 ``CREATE_SUSPENDED`` +
    ``STARTUPINFOEX``，而 ``subprocess.Popen`` 起完就把线程句柄关了，拿不到手。所以这条
    如实写在这里，也如实写进 ``isolation_report()['notes']``。
    """
    if sys.platform != "win32":
        return "非 win32"
    try:                                          # pragma: no cover - 平台分支
        k = _kernel32()
        # PROCESS_SET_QUOTA | PROCESS_TERMINATE：AssignProcessToJobObject 要的两项权限。
        handle = k.OpenProcess(0x0100 | 0x0001, False, int(pid))
        if not handle:
            return f"OpenProcess 失败 err={ctypes.get_last_error()}"
        try:
            if not k.AssignProcessToJobObject(job, handle):
                return f"AssignProcessToJobObject 失败 err={ctypes.get_last_error()}"
        finally:
            k.CloseHandle(handle)
        return ""
    except Exception as exc:                      # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def pid_alive(pid: int) -> bool:
    """这个 pid 现在还活着吗。杀完之后**回头确认**用的，不是装饰。

    win32 走 ``OpenProcess`` + ``GetExitCodeProcess``（退出码恰好是 259 的进程会被误判为
    存活，这是 Win32 API 本身的歧义，无解）；POSIX 走 ``kill(pid, 0)``（要求调用方已经
    ``wait()`` 收过尸，否则僵尸进程仍算存活）。
    """
    if not pid or pid < 0:
        return False
    if sys.platform == "win32":                   # pragma: no cover - 平台分支
        try:
            k = _kernel32()
            handle = k.OpenProcess(0x1000, False, int(pid))   # QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            try:
                code = wintypes.DWORD()
                if not k.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == _STILL_ACTIVE
            finally:
                k.CloseHandle(handle)
        except Exception:                         # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------- #
# POSIX：RLIMIT_AS
# --------------------------------------------------------------------------- #
def _rlimit_preexec(mem_limit_mb: int) -> Optional[Any]:
    """返回一个在子进程里设地址空间上限的 ``preexec_fn``；win32 上返回 None。"""
    if sys.platform == "win32":
        return None
    try:
        import resource                           # noqa: PLC0415 - POSIX 才有
    except Exception:                             # noqa: BLE001
        return None

    def _apply() -> None:                         # pragma: no cover - POSIX 分支
        try:
            cap = max(1, int(mem_limit_mb)) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (ValueError, OSError):
            pass

    return _apply


# --------------------------------------------------------------------------- #
# 一次执行的结果
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecResult:
    """一次子进程执行的全部可观察结果。

    ``outcome`` 是四选一：``ok``（退出码 0）/ ``nonzero``（跑完了但退出码非 0）/
    ``timeout``（墙钟到点被杀）/ ``error``（根本没起来：脚本不存在、OSError）。
    这四种在「没拿到想要的结果」上长得一样，但要改的东西完全不同，所以分开记。

    ``pid`` 不在最小契约里，多给一个：杀完之后要能**从外面核对**那个进程真的没了
    （见 :func:`pid_alive`），没有 pid 就只能相信自己。放在最后且有默认值，按位置构造
    前八个字段的调用方不受影响。
    """

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    command: list[str]
    outcome: str
    pid: int = 0


# --------------------------------------------------------------------------- #
# 沙箱
# --------------------------------------------------------------------------- #
class Sandbox:
    """在隔离的子进程里跑技能脚本与模型现写的 python。

    ``root`` 是技能根目录（例如 ``third_party/math-skill``）：``run_script`` 的相对路径
    按它解析，且校验解析结果必须仍在它之下。**脚本原地跑，不拷贝**——拷一份进临时目录
    会在 ``third_party/`` 之外多出一棵可写的树，而 cwd 已经是临时目录，写操作落不到原
    树上，拷贝买不到额外的安全。

    典型用法::

        with Sandbox(Path("third_party/math-skill"), timeout_s=20) as sb:
            r = sb.run_script("scripts/math_verify.py", ["--json", "equiv", "1/2", "0.5"])
    """

    def __init__(self, root: Any, *, timeout_s: float = 30.0, mem_limit_mb: int = 1024,
                 env_allow: Sequence[str] = (), python: str | None = None,
                 max_output_chars: int = 200_000) -> None:
        self.root = Path(root)
        self.timeout_s = float(timeout_s)
        self.mem_limit_mb = int(mem_limit_mb)
        self.env_allow = [k for k in env_allow if not _denied(k)]
        #: 被 ``_denied`` 拦下的那些名字。如实记着，供 isolation_report 交代。
        self.env_denied = [k for k in env_allow if _denied(k)]
        self.python = python or sys.executable
        self.max_output_chars = int(max_output_chars)

        self._workdir: Optional[Path] = None
        self._job: Optional[int] = None
        self._job_note = ""
        self._assign_failed = False
        self._closed = False
        self._counter = 0

    # ---- 生命周期 ------------------------------------------------------- #
    def __enter__(self) -> "Sandbox":
        self._ensure_ready()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """关掉 Job（连坐杀光残留进程）再删临时目录。顺序不能反。

        先删目录的话，残留的孙进程可能还攥着目录里的文件句柄，Windows 上 rmtree 直接
        ``PermissionError``，删剩半个目录还漏一个活进程。
        """
        if self._job is not None:                 # pragma: no cover - 平台分支
            try:
                k = _kernel32()
                k.TerminateJobObject(self._job, 1)
                k.CloseHandle(self._job)          # KILL_ON_JOB_CLOSE：补一道保险
            except Exception:                     # noqa: BLE001
                pass
            self._job = None
        if self._workdir is not None and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
        self._closed = True

    @property
    def workdir(self) -> Path:
        """本沙箱的临时工作目录（= 子进程的 cwd）。没建就现建。"""
        self._ensure_ready()
        assert self._workdir is not None
        return self._workdir

    def _ensure_ready(self) -> None:
        """懒建工作目录与 Job。不用 ``with`` 也能跑，只是没人替你收尾。"""
        if self._closed:
            raise SandboxError("沙箱已关闭，不能再执行")
        if self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="s2f_sandbox_"))
        if sys.platform == "win32" and self._job is None and not self._job_note:
            self._job, self._job_note = _win_create_job(self.mem_limit_mb)
            if self._job is None and not self._job_note:
                self._job_note = "未知原因"

    # ---- 隔离等级：如实交代 --------------------------------------------- #
    def isolation_report(self) -> dict:
        """这台沙箱**实际**做到了哪几条。给 experiment.py 印进报告用。

        ``mem_limit`` 是「真的施加上了」而不是「打算施加」：win32 上 Job 建不起来、或者
        任何一次挂载失败过，它就是 ``False``。``network_blocked`` 恒 ``False``。
        """
        self._ensure_ready()
        if sys.platform == "win32":
            mem_ok = self._job is not None and not self._assign_failed
            method = "job-object" if mem_ok else "none"
        else:
            mem_ok = _rlimit_preexec(self.mem_limit_mb) is not None
            method = "rlimit" if mem_ok else "none"
        notes = [
            "网络无法在 OS 层阻断：不上容器/网络命名空间就做不到，这里不假装做到了。",
            "只隔离 cwd：子进程仍能读整块盘，写操作也只是「没理由往外写」而非「不能」。",
        ]
        if sys.platform == "win32":
            notes.append("Job 是进程起来之后挂上的，这几十毫秒里派生的孙进程不受 Job 管。")
            if self._job is None:
                notes.append(f"Job Object 建不起来，内存上限已降级为无：{self._job_note}")
            elif self._assign_failed:
                notes.append("有进程挂进 Job 失败过，内存上限对这台沙箱不成立。")
        return {
            "platform": sys.platform,
            "timeout": True,
            "timeout_s": self.timeout_s,
            "kill_tree": True,
            "env_allowlist": True,
            "env_allow": list(self.env_allow),
            "env_denied": list(self.env_denied),
            "mem_limit": bool(mem_ok),
            "mem_limit_mb": self.mem_limit_mb,
            "mem_limit_method": method,
            "cwd_isolated": True,
            "network_blocked": False,
            "filesystem_isolated": False,
            "python": self.python,
            "notes": notes,
        }

    # ---- 环境 ----------------------------------------------------------- #
    def _child_env(self) -> dict:
        """**从空字典搭起**的子进程环境。父进程环境不是基底，是取用来源。"""
        env: dict[str, str] = {}
        for key in list(_BASE_ALLOW) + list(self.env_allow):
            if _denied(key):
                continue
            val = os.environ.get(key)
            if val is not None:
                env[key] = val
        env.setdefault("PATH", os.defpath)
        # Windows 上 python 的 stdout 默认走 ANSI 代码页，打印非 ASCII 直接
        # UnicodeEncodeError，外面看只是「输出为空」。POSIX 上给它不改变任何行为。
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        # third_party/ 是编译目标，必须逐字节不变——不许在它旁边落 __pycache__。
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        work = str(self.workdir)
        for key in ("TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE"):
            env[key] = work
        return env

    # ---- 执行 ----------------------------------------------------------- #
    def _resolve(self, script_rel: str) -> Path:
        """把相对路径解到 ``root`` 之下，逃逸即 :class:`SandboxError`。"""
        if not script_rel or not str(script_rel).strip():
            raise SandboxError("脚本路径为空")
        root = self.root.resolve()
        try:
            target = (root / str(script_rel)).resolve()
        except OSError as exc:
            raise SandboxError(f"脚本路径解析失败 {script_rel!r}: {exc}") from exc
        if target != root and not target.is_relative_to(root):
            raise SandboxError(f"脚本路径逃出技能根目录: {script_rel!r} → {target}")
        return target

    def run_script(self, script_rel: str, argv: Sequence[str]) -> ExecResult:
        """跑 ``root/<script_rel>``，参数 ``argv`` 原样传给它。

        注意 ``math_verify.py`` 的 ``--json`` 必须排在子命令**前面**：
        ``["--json", "equiv", a, b]`` 对，``["equiv", a, b, "--json"]`` 退出码 2。
        这一层不替调用方重排——猜参数顺序一旦猜错，错法比原样传更难查。
        """
        script = self._resolve(script_rel)
        cmd = [self.python, str(script), *[str(a) for a in argv]]
        if not script.is_file():
            return ExecResult(False, ERROR_RC, "", f"script not found: {script_rel}",
                              0.0, False, cmd, OUTCOME_ERROR)
        return self._spawn(cmd)

    def run_code(self, code: str, argv: Sequence[str] = ()) -> ExecResult:
        """把一段 python 落成工作目录里的临时文件再跑。模型现写的代码走这条。

        落文件而不是 ``python -c``：traceback 里才有真实行号，而调试模型写的代码，
        「第几行炸的」是最要紧的那一条信息。
        """
        self._ensure_ready()
        self._counter += 1
        # 不碰全局 random：conftest 用固定种子钉了别处的注入率，这里取一个数就会错位。
        path = self.workdir / f"_code_{os.getpid()}_{self._counter}.py"
        path.write_text(code, encoding="utf-8")
        cmd = [self.python, str(path), *[str(a) for a in argv]]
        return self._spawn(cmd)

    def _spawn(self, cmd: list[str]) -> ExecResult:
        """起进程、等墙钟、超时就杀树，最后收结果。"""
        self._ensure_ready()
        popen_kw: dict[str, Any] = {}
        if sys.platform == "win32":
            popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            # 自成进程组，这样 killpg 一刀能带走它派生的全部后代。
            popen_kw["start_new_session"] = True
            preexec = _rlimit_preexec(self.mem_limit_mb)
            if preexec is not None:
                popen_kw["preexec_fn"] = preexec

        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(self.workdir), env=self._child_env(),
                stdin=subprocess.DEVNULL,          # 读 stdin 的子进程不许把我们挂住
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", **popen_kw)
        except OSError as exc:
            return ExecResult(False, ERROR_RC, "", f"{type(exc).__name__}: {exc}",
                              time.monotonic() - t0, False, cmd, OUTCOME_ERROR)

        if self._job is not None:                  # pragma: no cover - 平台分支
            note = _win_assign_job(self._job, proc.pid)
            if note:
                self._assign_failed = True
                self._job_note = note

        try:
            out, err = proc.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            out, err = self._drain(proc)
            dur = time.monotonic() - t0
            alive = pid_alive(proc.pid)
            tail = "" if not alive else "（杀完之后它居然还活着，这条要当真事查）"
            return ExecResult(
                False, TIMEOUT_RC, self._clip(out),
                self._clip(f"timed out after {self.timeout_s}s{tail}\n{err}"),
                dur, True, cmd, OUTCOME_TIMEOUT, proc.pid)
        except Exception as exc:                   # noqa: BLE001 - 通信本身炸了
            self._kill_tree(proc)
            return ExecResult(False, ERROR_RC, "", f"{type(exc).__name__}: {exc}",
                              time.monotonic() - t0, False, cmd, OUTCOME_ERROR, proc.pid)

        rc = proc.returncode
        return ExecResult(rc == 0, rc, self._clip(out or ""), self._clip(err or ""),
                          time.monotonic() - t0, False, cmd,
                          OUTCOME_OK if rc == 0 else OUTCOME_NONZERO, proc.pid)

    def _kill_tree(self, proc: subprocess.Popen) -> None:
        """把整棵进程树杀干净。只 kill 父进程不够：孙进程会活下来接着烧 CPU。"""
        if sys.platform == "win32":                # pragma: no cover - 平台分支
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=15,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            except Exception:                      # noqa: BLE001
                pass
            if self._job is not None:              # Job 里的残留一并带走
                try:
                    _kernel32().TerminateJobObject(self._job, 1)
                except Exception:                  # noqa: BLE001
                    pass
        else:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)                  # 收尸，pid_alive 才问得出真话
        except Exception:                          # noqa: BLE001
            pass

    @staticmethod
    def _drain(proc: subprocess.Popen) -> tuple[str, str]:
        """杀完之后把管道里剩下的读干净。

        ``TimeoutExpired`` 本身**不带**已经产出的输出（win32 上由读取线程持有），必须在
        杀掉之后再 communicate 一次才拿得到——不然一次超时在下游看起来就是「没有输出」。
        """
        try:
            out, err = proc.communicate(timeout=10)
            return out or "", err or ""
        except Exception:                          # noqa: BLE001
            return "", ""

    def _clip(self, text: str) -> str:
        """输出截断。模型写的循环 print 能刷出几百 MB，整条带进内存没有意义。"""
        if len(text) <= self.max_output_chars:
            return text
        return text[:self.max_output_chars] + f"\n...[截断，原长 {len(text)} 字符]"

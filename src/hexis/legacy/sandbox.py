"""A sandbox for running real code: a skill's own ``scripts/*.py`` and python the model writes on the fly both run here.

Compiling a skill cannot avoid **actually running its scripts**. A math skill's ``scripts/math_verify.py`` is 788
lines of sympy in which ``parse_expr`` goes all the way down to python's ``eval``, and the expressions fed into it come
from a large model. This layer is the **only** boundary in the whole chain, so it is written as a boundary, not as a
tool.

Four isolation measures that really are in place, plus one that is not and has to be stated as such:

* **Wall-clock timeout + killing the process tree.** ``equiv "9^9^9" "1"`` never returns on this machine: that script
  has no internal timeout anywhere. Killing only the parent is not enough, because code the model writes spawns child
  processes. win32 uses ``taskkill /F /T`` followed by ``TerminateJobObject`` for good measure, POSIX uses ``killpg``
  on the process group; after the kill it checks that the processes are really gone (:func:`pid_alive`), and when
  that cannot be confirmed it does not claim otherwise.
* **Environment starvation.** The child's environment is **built from an empty dict** holding only the allowlisted
  names. ``API_KEY`` / ``ANTHROPIC_*`` / ``OPENAI_*`` / ``HTTP(S)_PROXY`` are withheld even when the caller names them
  explicitly in ``env_allow`` (:func:`_denied`): a single slip would put a live production key into python an LLM
  just wrote.
* **Memory limit.** ``setrlimit(RLIMIT_AS)`` on POSIX; a Job Object on win32 (``CreateJobObjectW`` +
  ``ProcessMemoryLimit`` + ``KILL_ON_JOB_CLOSE``). If the Job cannot be created or the process cannot be assigned to
  it, the sandbox **degrades and reports it honestly**: in :meth:`Sandbox.isolation_report` ``mem_limit`` becomes
  ``False`` and ``mem_limit_method`` becomes ``"none"``. Nothing is claimed that was not done.
* **cwd isolation.** Every sandbox gets its own temporary working directory, never the repository; scripts are
  resolved against ``root`` and checked so they cannot escape with ``..`` (an escape raises :class:`SandboxError`
  right away instead of quietly returning a failure). ``PYTHONDONTWRITEBYTECODE=1`` additionally guarantees that no
  ``__pycache__`` gets written into ``third_party/``: that tree is a compilation target and must stay byte-for-byte
  unchanged.

* **Network: not blocked.** Without containers or network namespaces there is no way to cut a child process off the
  network at the OS level, and this module does not pretend it can. ``isolation_report()["network_blocked"]`` is
  always ``False``, so that the experiment report prints the **real** isolation level rather than a flattering one.
  Likewise ``filesystem_isolated`` is ``False``: only the cwd changes, and the child can still read the whole disk.

Two calling conventions were learned the hard way; they are written down here so nobody trips over them again:

* The ``--json`` of ``math_verify.py`` belongs to the **top-level** parser and must come **before** the subcommand.
  ``--json equiv a b`` exits with 0; ``equiv a b --json`` exits with 2 (unrecognized arguments).
* Without ``PYTHONIOENCODING=utf-8``, python on Windows raises ``UnicodeEncodeError`` when printing non-ASCII, which
  from outside only looks like "empty output": a phantom failure that is practically impossible to track down.

**One sandbox per run**: step three has to read intermediate files left behind by step one, which a fresh sandbox per
step would make impossible. For the same reason this class is **not thread-safe**: the win32 tree kill takes down
every process in the same Job, so two commands running at once in one sandbox would kill each other.
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

#: The few names passed through from the parent environment **by name**. Nothing else enters the child.
_BASE_ALLOW = (
    "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "SYSTEMDRIVE",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "LANG", "LC_ALL", "TZ",
)

#: Never passed through, however the caller names them. Any name containing one of these substrings is withheld.
_DENY_SUBSTR = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL")
_DENY_PREFIX = ("ANTHROPIC_", "OPENAI_", "AWS_", "AZURE_", "GOOGLE_", "GCP_",
                "HF_")
_DENY_EXACT = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY", "NO_PROXY")


class SandboxError(RuntimeError):
    """The sandbox refuses to execute: path escapes, a closed sandbox and similar **caller errors**.

    Deliberately kept apart from "the script crashed": the latter is a piece of data in :class:`ExecResult` with
    ``outcome='error'``, the former is an exception. The reasoning: when whatever a guard raises is treated as
    ``False``, the machine quietly takes the wrong edge and in the end reports only "the result is wrong".
    """


def _denied(name: str) -> bool:
    """Whether this environment variable name belongs to the "withheld even when named" class."""
    up = name.upper()
    return (up in _DENY_EXACT
            or any(up.startswith(p) for p in _DENY_PREFIX)
            or any(s in up for s in _DENY_SUBSTR))


# --------------------------------------------------------------------------- #
# win32: Job Object (memory limit + everything killed when it is closed)
# --------------------------------------------------------------------------- #
_JOB_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_EXTENDED_LIMIT_CLASS = 9
_STILL_ACTIVE = 259

if sys.platform == "win32":                       # pragma: no cover - platform branch
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
        # restype must be explicit: the default c_int truncates 64-bit handles to 32 bits (looks valid, fails when used).
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
    """Create a Job with a per-process memory limit that kills everything when its handle closes. ``(None, reason)`` on failure."""
    if sys.platform != "win32":
        return None, "not win32"
    try:                                          # pragma: no cover - platform branch
        k = _kernel32()
        handle = k.CreateJobObjectW(None, None)
        if not handle:
            return None, f"CreateJobObjectW failed err={ctypes.get_last_error()}"
        info = _JOB_EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_LIMIT_PROCESS_MEMORY | _JOB_LIMIT_KILL_ON_JOB_CLOSE)
        info.ProcessMemoryLimit = max(1, int(mem_limit_mb)) * 1024 * 1024
        ok = k.SetInformationJobObject(handle, _JOB_EXTENDED_LIMIT_CLASS,
                                       ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            err = ctypes.get_last_error()
            k.CloseHandle(handle)
            return None, f"SetInformationJobObject failed err={err}"
        return int(handle), ""
    except Exception as exc:                      # noqa: BLE001 - degrade, do not crash
        return None, f"{type(exc).__name__}: {exc}"


def _win_assign_job(job: int, pid: int) -> str:
    """Assign an already started process to the Job. Returns an empty string on success, otherwise the failure reason.

    There is a **real race** here: the process starts first and is assigned to the Job afterwards, and grandchildren it
    spawns during those few tens of milliseconds (python interpreter startup) are not governed by the Job. Closing the
    gap completely would take ``CREATE_SUSPENDED`` + ``STARTUPINFOEX``, but ``subprocess.Popen`` closes the thread
    handle right after starting, so it is out of reach. That is why this is stated honestly here, and just as honestly
    in ``isolation_report()['notes']``.
    """
    if sys.platform != "win32":
        return "not win32"
    try:                                          # pragma: no cover - platform branch
        k = _kernel32()
        # PROCESS_SET_QUOTA | PROCESS_TERMINATE: the two access rights AssignProcessToJobObject needs.
        handle = k.OpenProcess(0x0100 | 0x0001, False, int(pid))
        if not handle:
            return f"OpenProcess failed err={ctypes.get_last_error()}"
        try:
            if not k.AssignProcessToJobObject(job, handle):
                return f"AssignProcessToJobObject failed err={ctypes.get_last_error()}"
        finally:
            k.CloseHandle(handle)
        return ""
    except Exception as exc:                      # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def pid_alive(pid: int) -> bool:
    """Is this pid still alive right now? Used to **double-check** after a kill; it is not decoration.

    win32 uses ``OpenProcess`` + ``GetExitCodeProcess`` (a process whose exit code happens to be 259 is misjudged as
    alive; that ambiguity is inherent in the Win32 API and cannot be resolved); POSIX uses ``kill(pid, 0)`` (the caller
    must already have reaped the process with ``wait()``, otherwise a zombie still counts as alive).
    """
    if not pid or pid < 0:
        return False
    if sys.platform == "win32":                   # pragma: no cover - platform branch
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
# POSIX: RLIMIT_AS
# --------------------------------------------------------------------------- #
def _rlimit_preexec(mem_limit_mb: int) -> Optional[Any]:
    """Return a ``preexec_fn`` that sets the address-space limit in the child; returns None on win32."""
    if sys.platform == "win32":
        return None
    try:
        import resource  # noqa: PLC0415 - POSIX only
    except Exception:                             # noqa: BLE001
        return None

    def _apply() -> None:                         # pragma: no cover - POSIX branch
        try:
            cap = max(1, int(mem_limit_mb)) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (ValueError, OSError):
            pass

    return _apply


# --------------------------------------------------------------------------- #
# Result of one execution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecResult:
    """Everything observable about one child-process execution.

    ``outcome`` is one of four: ``ok`` (exit code 0) / ``nonzero`` (ran to the end but with a non-zero exit code) /
    ``timeout`` (killed when the wall clock ran out) / ``error`` (never started: missing script, OSError).
    All four look the same as far as "did not get the wanted result" goes, but what needs fixing is completely
    different, so they are recorded separately.

    ``pid`` is not part of the minimal contract; it is an extra: after a kill it must be possible to **verify from the
    outside** that the process is really gone (see :func:`pid_alive`); without a pid we could only trust ourselves. It
    comes last and has a default, so callers that build the first eight fields positionally are unaffected.
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
# Sandbox
# --------------------------------------------------------------------------- #
class Sandbox:
    """Run skill scripts and python the model writes on the fly in isolated child processes.

    ``root`` is the skill root directory (for example ``third_party/math-skill``): relative paths given to
    ``run_script`` are resolved against it, and the resolved path is checked to still lie under it. **Scripts run in
    place, without copying**: a copy in a temporary directory would add a writable tree outside ``third_party/``, and
    since the cwd is already a temporary directory, writes do not land in the original tree anyway; copying buys no
    extra safety.

    Typical usage::

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
        #: The names blocked by ``_denied``. Recorded honestly, so that isolation_report can disclose them.
        self.env_denied = [k for k in env_allow if _denied(k)]
        self.python = python or sys.executable
        self.max_output_chars = int(max_output_chars)

        self._workdir: Optional[Path] = None
        self._job: Optional[int] = None
        self._job_note = ""
        self._assign_failed = False
        self._closed = False
        self._counter = 0

    # ---- lifecycle ----------------------------------------------------- #
    def __enter__(self) -> "Sandbox":
        self._ensure_ready()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the Job (killing every leftover process with it), then delete the temporary directory. Never the other way round.

        If the directory went first, leftover grandchildren might still hold file handles inside it; on Windows rmtree
        then fails with ``PermissionError``, leaving half a directory behind and a live process on the loose.
        """
        if self._job is not None:                 # pragma: no cover - platform branch
            try:
                k = _kernel32()
                k.TerminateJobObject(self._job, 1)
                k.CloseHandle(self._job)          # KILL_ON_JOB_CLOSE: one more safeguard
            except Exception:                     # noqa: BLE001
                pass
            self._job = None
        if self._workdir is not None and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
        self._closed = True

    @property
    def workdir(self) -> Path:
        """This sandbox's temporary working directory (= the child's cwd). Created on demand."""
        self._ensure_ready()
        assert self._workdir is not None
        return self._workdir

    def _ensure_ready(self) -> None:
        """Lazily create the working directory and the Job. Works without ``with`` too; there is just nobody to clean up after you."""
        if self._closed:
            raise SandboxError("sandbox is closed and cannot execute anything")
        if self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="s2f_sandbox_"))
        if sys.platform == "win32" and self._job is None and not self._job_note:
            self._job, self._job_note = _win_create_job(self.mem_limit_mb)
            if self._job is None and not self._job_note:
                self._job_note = "unknown reason"

    # ---- isolation level, disclosed honestly ------------------------- #
    def isolation_report(self) -> dict:
        """Which of the measures this sandbox **actually** achieved, for the experiment report to print.

        ``mem_limit`` means "really applied", not "intended": on win32, if the Job could not be created or any
        assignment ever failed, it is ``False``. ``network_blocked`` is always ``False``.
        """
        self._ensure_ready()
        if sys.platform == "win32":
            mem_ok = self._job is not None and not self._assign_failed
            method = "job-object" if mem_ok else "none"
        else:
            mem_ok = _rlimit_preexec(self.mem_limit_mb) is not None
            method = "rlimit" if mem_ok else "none"
        notes = [
            "the network cannot be blocked at the OS level: impossible without containers/network namespaces, and this sandbox does not pretend otherwise",
            "only the cwd is isolated: the child can still read the whole disk, and it merely has \"no reason to write elsewhere\" rather than being unable to",
        ]
        if sys.platform == "win32":
            notes.append("the Job is attached after the process starts; grandchildren spawned in those few tens of milliseconds are not governed by it")
            if self._job is None:
                notes.append(f"the Job Object could not be created, so the memory limit degraded to none: {self._job_note}")
            elif self._assign_failed:
                notes.append("assigning a process to the Job failed at least once, so the memory limit does not hold for this sandbox")
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

    # ---- environment --------------------------------------------------- #
    def _child_env(self) -> dict:
        """The child environment, **built from an empty dict**. The parent environment is not the base, only a source to draw from."""
        env: dict[str, str] = {}
        for key in list(_BASE_ALLOW) + list(self.env_allow):
            if _denied(key):
                continue
            val = os.environ.get(key)
            if val is not None:
                env[key] = val
        env.setdefault("PATH", os.defpath)
        # On Windows python's stdout defaults to the ANSI code page, so printing non-ASCII raises
        # UnicodeEncodeError, which from outside only looks like "empty output". On POSIX setting it changes nothing.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        # third_party/ is a compilation target and must stay byte-for-byte unchanged: no __pycache__ may land next to it.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        work = str(self.workdir)
        for key in ("TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE"):
            env[key] = work
        return env

    # ---- execution ----------------------------------------------------- #
    def _resolve(self, script_rel: str) -> Path:
        """Resolve a relative path under ``root``; an escape raises :class:`SandboxError`."""
        if not script_rel or not str(script_rel).strip():
            raise SandboxError("script path is empty")
        root = self.root.resolve()
        try:
            target = (root / str(script_rel)).resolve()
        except OSError as exc:
            raise SandboxError(f"cannot resolve script path {script_rel!r}: {exc}") from exc
        if target != root and not target.is_relative_to(root):
            raise SandboxError(f"script path escapes the skill root: {script_rel!r} → {target}")
        return target

    def run_script(self, script_rel: str, argv: Sequence[str]) -> ExecResult:
        """Run ``root/<script_rel>``, passing ``argv`` to it unchanged.

        Note that the ``--json`` of ``math_verify.py`` must come **before** the subcommand:
        ``["--json", "equiv", a, b]`` is right, ``["equiv", a, b, "--json"]`` exits with 2.
        This layer does not reorder arguments for the caller: a wrong guess about argument order fails in ways that are
        harder to track down than passing them through unchanged.
        """
        script = self._resolve(script_rel)
        cmd = [self.python, str(script), *[str(a) for a in argv]]
        if not script.is_file():
            return ExecResult(False, ERROR_RC, "", f"script not found: {script_rel}",
                              0.0, False, cmd, OUTCOME_ERROR)
        return self._spawn(cmd)

    def run_code(self, code: str, argv: Sequence[str] = ()) -> ExecResult:
        """Write a piece of python to a temporary file in the working directory, then run it. Model-written code goes this way.

        A file rather than ``python -c``: only then does the traceback carry real line numbers, and when debugging code
        a model wrote, "which line blew up" is the single most important piece of information.
        """
        self._ensure_ready()
        self._counter += 1
        # leave the global random alone: conftest pins injection rates elsewhere with a fixed seed; one draw here shifts them.
        path = self.workdir / f"_code_{os.getpid()}_{self._counter}.py"
        path.write_text(code, encoding="utf-8")
        cmd = [self.python, str(path), *[str(a) for a in argv]]
        return self._spawn(cmd)

    def _spawn(self, cmd: list[str]) -> ExecResult:
        """Start the process, wait on the wall clock, kill the tree on timeout, then collect the result."""
        self._ensure_ready()
        popen_kw: dict[str, Any] = {}
        if sys.platform == "win32":
            popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            # its own process group, so that a single killpg takes all of its descendants along.
            popen_kw["start_new_session"] = True
            preexec = _rlimit_preexec(self.mem_limit_mb)
            if preexec is not None:
                popen_kw["preexec_fn"] = preexec

        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(self.workdir), env=self._child_env(),
                stdin=subprocess.DEVNULL,          # a child reading stdin must not hang us
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", **popen_kw)
        except OSError as exc:
            return ExecResult(False, ERROR_RC, "", f"{type(exc).__name__}: {exc}",
                              time.monotonic() - t0, False, cmd, OUTCOME_ERROR)

        if self._job is not None:                  # pragma: no cover - platform branch
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
            tail = "" if not alive else " (still alive after the kill; investigate this as a real problem)"
            return ExecResult(
                False, TIMEOUT_RC, self._clip(out),
                self._clip(f"timed out after {self.timeout_s}s{tail}\n{err}"),
                dur, True, cmd, OUTCOME_TIMEOUT, proc.pid)
        except Exception as exc:                   # noqa: BLE001 - the communication itself failed
            self._kill_tree(proc)
            return ExecResult(False, ERROR_RC, "", f"{type(exc).__name__}: {exc}",
                              time.monotonic() - t0, False, cmd, OUTCOME_ERROR, proc.pid)

        rc = proc.returncode
        return ExecResult(rc == 0, rc, self._clip(out or ""), self._clip(err or ""),
                          time.monotonic() - t0, False, cmd,
                          OUTCOME_OK if rc == 0 else OUTCOME_NONZERO, proc.pid)

    def _kill_tree(self, proc: subprocess.Popen) -> None:
        """Kill the whole process tree. Killing only the parent is not enough: grandchildren survive and keep burning CPU."""
        if sys.platform == "win32":                # pragma: no cover - platform branch
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=15,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            except Exception:                      # noqa: BLE001
                pass
            if self._job is not None:              # take the leftovers in the Job along too
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
            proc.wait(timeout=10)                  # reap it, so that pid_alive tells the truth
        except Exception:                          # noqa: BLE001
            pass

    @staticmethod
    def _drain(proc: subprocess.Popen) -> tuple[str, str]:
        """After the kill, drain whatever is left in the pipes.

        ``TimeoutExpired`` itself does **not** carry the output produced so far (on win32 the reader threads hold it);
        only one more communicate after the kill gets it. Otherwise a timeout looks like "no output" downstream.
        """
        try:
            out, err = proc.communicate(timeout=10)
            return out or "", err or ""
        except Exception:                          # noqa: BLE001
            return "", ""

    def _clip(self, text: str) -> str:
        """Truncate output. A print loop in model-written code can emit hundreds of MB; keeping all of it in memory is pointless."""
        if len(text) <= self.max_output_chars:
            return text
        return text[:self.max_output_chars] + f"\n...[truncated, original length {len(text)} characters]"

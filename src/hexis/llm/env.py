"""`.env` discovery and parsing, and the three OpenAI-compatible settings (MODEL / BASE_URL / API_KEY).

Real experiments connect to live model endpoints, and the configuration lives in the `.env` at the repository
root. This module does only two small things: **find that file** and **turn it into a configuration object**.

**Why "walk up to the directory holding .git" instead of counting ``.parent`` hops.** A fixed number of hops
breaks every time the directory layout changes. So we search for ``.env`` upward level by level from where this
file is, and stop at the directory that holds ``.git`` -- stopping at the checkout boundary means a ``.env`` in
some user directory outside the checkout is never picked up by mistake.

**API keys never leave memory.** This module (and anyone holding an :class:`LLMConfig`) never prints, logs or
writes the API key to disk: ``LLMConfig``'s repr does not contain it, and :meth:`LLMConfig.redacted` only shows
the last 4 characters. When a key is missing, the error names "which key is missing" and carries no values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, MutableMapping, Optional

#: key names of the three settings. The order is the order in which errors name them.
REQUIRED_KEYS = ("MODEL", "BASE_URL", "API_KEY")

ENV_FILENAME = ".env"


class EnvError(RuntimeError):
    """Configuration is missing or unusable. Messages contain only key names and paths, never key values."""


# --------------------------------------------------------------------------- #
# Finding the file
# --------------------------------------------------------------------------- #
def find_env_file(start: Optional[Path] = None) -> Optional[Path]:
    """Search upward from ``start`` (default: where this file is) for ``.env``, up to the directory holding ``.git``.

    Within one directory ``.env`` is checked before ``.git`` -- so the ``.env`` at the repository root is found;
    if none is found, the search stops at the checkout boundary and returns ``None`` instead of climbing further
    and picking up another project's configuration.
    """
    base = Path(start).resolve() if start is not None else Path(__file__).resolve()
    here = base if base.is_dir() else base.parent
    for d in (here, *here.parents):
        cand = d / ENV_FILENAME
        if cand.exists():
            return cand
        if (d / ".git").exists():        # checkout boundary: stop here
            break
    return None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` text into a dict. Blank lines, ``#`` comment lines and lines without ``=`` are skipped."""
    parsed: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        parsed[k.strip()] = v.strip().strip('"').strip("'")
    return parsed


def load_env(*, path: Optional[Path] = None, override: bool = False,
             environ: Optional[MutableMapping[str, str]] = None) -> dict[str, str]:
    """Read ``.env`` and add it to the environment variables (existing ones are not overridden by default); returns the parsed dict.

    Without ``path``, :func:`find_env_file` searches first from the current working directory, then from the
    package location; if nothing is found an empty dict is returned (not an error -- on CI, injecting real
    environment variables is normal). Without ``environ`` it is ``os.environ`` (tests can pass a temporary dict
    to avoid polluting the process environment). Idempotent: repeated calls do not change the result.
    """
    target = Path(path) if path is not None else (find_env_file(Path.cwd()) or find_env_file())
    if target is None or not target.exists():
        return {}
    parsed = parse_env_text(target.read_text(encoding="utf-8"))
    env = os.environ if environ is None else environ
    for k, v in parsed.items():
        if override or k not in env:
            env[k] = v
    return parsed


# --------------------------------------------------------------------------- #
# Configuration object
# --------------------------------------------------------------------------- #
def redact(secret: str) -> str:
    """Displayable form of an API key: only the last 4 characters. An empty string gives ``'<unset>'``.

    The mask prefix is ASCII ``***``: this string goes to Windows consoles and log files, so do not bet on a
    non-ASCII character surviving their encoding just for looks.
    """
    s = secret or ""
    if not s:
        return "<unset>"
    return "***" + s[-4:] if len(s) > 4 else "***"


@dataclass(frozen=True)
class LLMConfig:
    """The three settings of an OpenAI-compatible endpoint. ``api_key`` is **not in the repr** (dataclass ``repr=False``).

    Use :meth:`redacted` for display; logs, traces and error messages may only ever contain that.
    """

    model: str
    base_url: str
    api_key: str = field(repr=False, default="")

    @property
    def configured(self) -> bool:
        return bool(self.model and self.base_url and self.api_key)

    def redacted(self) -> str:
        """A one-line summary with the API key masked. Safe to write to logs."""
        return f"model={self.model} base_url={self.base_url} api_key={redact(self.api_key)}"


#: endpoint profiles: ``--provider minimax`` reads ``MINIMAX_API_KEY`` / ``MINIMAX_MODEL`` / ``MINIMAX_BASE_URL``,
#: and the latter two default to the values here. The model and endpoint are the ones used for harness calibration.
PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "minimax": {"MODEL": "MiniMax-M2.5-highspeed", "BASE_URL": "https://api.minimaxi.com/v1"},
    "deepseek": {"MODEL": "deepseek-v4-flash", "BASE_URL": "https://api.deepseek.com"},
}


def llm_config(*, environ: Optional[Mapping[str, str]] = None, profile: str = "", model: str = "",
               base_url: str = "", api_key_env: str = "") -> LLMConfig:
    """Get the three settings from the environment (reading ``.env`` first if needed), naming whichever is missing.

    If ``environ`` is given explicitly, ``.env`` is **not** read and ``os.environ`` is not touched -- tests and
    setups with several endpoints use it to pin the configuration source. If ``profile`` is given, the prefixed
    settings are read (``MINIMAX_API_KEY`` ...), with model / base_url defaulting to :data:`PROFILE_DEFAULTS`; the
    API key has no default.

    ``model`` and ``base_url`` override the values from the environment, and ``api_key_env`` names the variable
    that holds the key instead of ``API_KEY`` / ``<PROFILE>_API_KEY``. Errors name only the missing settings.
    """
    if environ is None:
        load_env()                       # idempotent; does not override existing environment variables
        env: Mapping[str, str] = os.environ
    else:
        env = environ
    if profile:
        pf = profile.strip().lower()
        pre = pf.upper() + "_"
        dflt = PROFILE_DEFAULTS.get(pf, {})
        key_name = api_key_env or pre + "API_KEY"
        model = (model or env.get(pre + "MODEL") or dflt.get("MODEL") or "").strip()
        base = (base_url or env.get(pre + "BASE_URL") or dflt.get("BASE_URL") or "").strip()
        key = (env.get(key_name) or "").strip()
        need = [n for n, v in ((pre + "MODEL", model), (pre + "BASE_URL", base),
                               (key_name, key)) if not v]
        if need:
            where = find_env_file()
            loc = str(where) if where else f"(no {ENV_FILENAME} found)"
            raise EnvError(f"endpoint profile {pf} is missing {', '.join(need)}; set the missing keys in {loc} or in the environment "
                           f"(this message does not print any values)")
        return LLMConfig(model=model, base_url=base, api_key=key)
    key_name = api_key_env or "API_KEY"
    settings = {"MODEL": (model or env.get("MODEL") or "").strip(),
                "BASE_URL": (base_url or env.get("BASE_URL") or "").strip(),
                key_name: (env.get(key_name) or "").strip()}
    missing = [k for k in ("MODEL", "BASE_URL", key_name) if not settings[k]]
    if missing:
        where = find_env_file()
        loc = str(where) if where else f"(no {ENV_FILENAME} found)"
        raise EnvError(
            f"LLM configuration is missing {', '.join(missing)}; set the missing keys in {loc} or in the environment "
            f"(this message does not print any values)")
    return LLMConfig(model=settings["MODEL"], base_url=settings["BASE_URL"], api_key=settings[key_name])


def endpoint_record(cfg: LLMConfig, provider: str = "") -> dict:
    """Provider, model and base URL for logs and manifests: no key, no credentials or query in the URL."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(cfg.base_url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    clean = urlunsplit((parts.scheme, host, parts.path, "", "")) if parts.scheme else cfg.base_url.split("?")[0]
    return {"provider": provider or "default", "model": cfg.model, "base_url": clean}

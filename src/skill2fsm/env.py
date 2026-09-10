"""`.env` 定位、解析，以及 OpenAI 兼容三件套（MODEL / BASE_URL / API_KEY）。

真实实验要连活的模型端点，配置就落在仓库根的 `.env` 里。这里只做两件小事：**找到那个
文件**、**把它变成配置对象**。

**为什么是「向上走到 .git 那一层」而不是数几个 ``.parent``。** 固定的跳数每次目录搬家都
会失效。
所以从自己所在位置逐级向上找 ``.env``，撞到拿着 ``.git`` 的那一层就停——停在检出边界上，
checkout 之外某个用户目录里的 ``.env`` 永远不会被误捡进来。

**密钥不落地。** 本模块（以及拿到 :class:`LLMConfig` 的任何人）绝不打印、记录、写盘
API key：``LLMConfig`` 的 repr 里没有它，:meth:`LLMConfig.redacted` 只给末 4 位。缺 key
时的报错点名「缺哪个键」，不带任何取值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, MutableMapping, Optional

#: 三件套的键名。顺序即报错里的点名顺序。
REQUIRED_KEYS = ("MODEL", "BASE_URL", "API_KEY")

ENV_FILENAME = ".env"


class EnvError(RuntimeError):
    """配置缺失或不可用。消息里只出现键名与路径，绝不出现键值。"""


# --------------------------------------------------------------------------- #
# 找文件
# --------------------------------------------------------------------------- #
def find_env_file(start: Optional[Path] = None) -> Optional[Path]:
    """从 ``start``（默认本文件所在处）逐级向上找 ``.env``，到 ``.git`` 那一层为止。

    同一层里 ``.env`` 先于 ``.git`` 判定——仓库根上的 ``.env`` 因此能被找到；找不到就
    在检出边界上收手，返回 ``None``，而不是继续往上捡到别的项目的配置。
    """
    base = Path(start).resolve() if start is not None else Path(__file__).resolve()
    here = base if base.is_dir() else base.parent
    for d in (here, *here.parents):
        cand = d / ENV_FILENAME
        if cand.exists():
            return cand
        if (d / ".git").exists():        # 检出边界：到此为止
            break
    return None


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def parse_env_text(text: str) -> dict[str, str]:
    """把 ``KEY=VALUE`` 文本解析成 dict。空行、``#`` 注释行、没有 ``=`` 的行一律跳过。"""
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
    """读 ``.env`` 并补进环境变量（默认不覆盖已有的），返回解析出的 dict。

    ``path`` 不给就先从当前工作目录、再从包所在位置用 :func:`find_env_file` 找；找不到就返回空 dict（不是错误——CI 上靠
    真实环境变量注入是正常的）。``environ`` 不给就是 ``os.environ``（测试可传一个临时
    dict，避免污染进程环境）。幂等：重复调用不改变结果。
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
# 配置对象
# --------------------------------------------------------------------------- #
def redact(secret: str) -> str:
    """密钥的可展示形式：只留末 4 位。空串给 ``'<unset>'``。

    打码前缀用 ASCII 的 ``***``：这行字要进 Windows 控制台和日志文件，别为了好看
    赌一个非 ASCII 字符能被那边的编码放过。
    """
    s = secret or ""
    if not s:
        return "<unset>"
    return "***" + s[-4:] if len(s) > 4 else "***"


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI 兼容端点的三件套。``api_key`` **不进 repr**（dataclass 的 ``repr=False``）。

    要展示时用 :meth:`redacted`；日志、轨迹、报错里一律只能出现它。
    """

    model: str
    base_url: str
    api_key: str = field(repr=False, default="")

    @property
    def configured(self) -> bool:
        return bool(self.model and self.base_url and self.api_key)

    def redacted(self) -> str:
        """一行摘要，密钥已打码。可以安全地写进日志。"""
        return f"model={self.model} base_url={self.base_url} api_key={redact(self.api_key)}"


#: 端点档案：``--provider minimax`` 读 ``MINIMAX_API_KEY`` / ``MINIMAX_MODEL`` / ``MINIMAX_BASE_URL``，
#: 后两者缺省用这里的值。模型与端点取 docs/HARNESS_CALIBRATION.md 标定过的那一套。
PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "minimax": {"MODEL": "MiniMax-M2.5-highspeed", "BASE_URL": "https://api.minimaxi.com/v1"},
    "deepseek": {"MODEL": "deepseek-v4-flash", "BASE_URL": "https://api.deepseek.com"},
}


def llm_config(*, environ: Optional[Mapping[str, str]] = None, profile: str = "") -> LLMConfig:
    """从环境（必要时先读 ``.env``）取三件套，缺哪个就点名哪个。

    ``environ`` 显式给了就**不**再读 ``.env``、也不碰 ``os.environ``——测试与多端点并存
    时靠它把配置来源钉死。``profile`` 给了就读带前缀的三件套（``MINIMAX_API_KEY`` …），
    model / base_url 缺省取 :data:`PROFILE_DEFAULTS`；密钥没有缺省。
    """
    if environ is None:
        load_env()                       # 幂等；不覆盖已有的环境变量
        env: Mapping[str, str] = os.environ
    else:
        env = environ
    if profile:
        pf = profile.strip().lower()
        pre = pf.upper() + "_"
        dflt = PROFILE_DEFAULTS.get(pf, {})
        model = (env.get(pre + "MODEL") or dflt.get("MODEL") or "").strip()
        base = (env.get(pre + "BASE_URL") or dflt.get("BASE_URL") or "").strip()
        key = (env.get(pre + "API_KEY") or "").strip()
        need = [n for n, v in ((pre + "MODEL", model), (pre + "BASE_URL", base),
                               (pre + "API_KEY", key)) if not v]
        if need:
            where = find_env_file()
            loc = str(where) if where else f"（没找到 {ENV_FILENAME}）"
            raise EnvError(f"端点档案 {pf} 缺少 {', '.join(need)}：请在 {loc} 或环境变量里设置。"
                           f"（本消息不打印任何键值）")
        return LLMConfig(model=model, base_url=base, api_key=key)
    missing = [k for k in REQUIRED_KEYS if not (env.get(k) or "").strip()]
    if missing:
        where = find_env_file()
        loc = str(where) if where else f"（没找到 {ENV_FILENAME}）"
        raise EnvError(
            f"LLM 配置缺少 {', '.join(missing)}：请在 {loc} 或环境变量里设置。"
            f"（本消息不打印任何键值）")
    return LLMConfig(model=env["MODEL"].strip(), base_url=env["BASE_URL"].strip(),
                     api_key=env["API_KEY"].strip())

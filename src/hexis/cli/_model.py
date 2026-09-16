"""Model endpoint options shared by the commands that call a model.

Every setting can come from a flag, from a named profile in the environment, or from the default environment
variables, in that order of precedence:

    --model        <PROFILE>_MODEL     MODEL
    --base-url     <PROFILE>_BASE_URL  BASE_URL
    --api-key-env  names the variable holding the key (default <PROFILE>_API_KEY or API_KEY)

``--provider NAME`` selects the profile. Variables may also be set in a ``.env`` file (see ``.env.example``);
variables already present in the environment take precedence. The API key is never accepted on the command line.
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from typing import Any, Iterator, Optional


def add_endpoint_args(ap: argparse.ArgumentParser, *, skip: tuple = (), max_tokens: int = 16384,
                      timeout: float = 180.0, retries: int = 6) -> None:
    """Add the endpoint options that the parser does not define yet (names listed in ``skip`` are left out)."""
    g = ap.add_argument_group("model endpoint")
    spec = [
        ("provider", dict(default="default", help="endpoint profile: 'default' reads MODEL / BASE_URL / API_KEY; "
                                                  "NAME reads NAME_MODEL / NAME_BASE_URL / NAME_API_KEY")),
        ("model", dict(default="", help="model id (overrides MODEL or <PROFILE>_MODEL)")),
        ("base-url", dict(default="", help="base URL of an OpenAI-compatible endpoint, e.g. https://host/v1 "
                                           "(overrides BASE_URL or <PROFILE>_BASE_URL)")),
        ("api-key-env", dict(default="", help="name of the environment variable that holds the API key "
                                              "(default API_KEY or <PROFILE>_API_KEY)")),
        ("temperature", dict(type=float, default=0.0, help="sampling temperature")),
        ("max-tokens", dict(type=int, default=max_tokens, help="maximum tokens per model response")),
        ("llm-timeout", dict(type=float, default=timeout, help="timeout of one model request in seconds")),
        ("llm-retries", dict(type=int, default=retries, help="attempts per model request (429 / 5xx / network errors)")),
        ("stream", dict(action="store_true", help="stream responses (avoids gateway timeouts on long generations)")),
        ("extra-body", dict(default="", help="JSON object merged into every request body, "
                                             "e.g. '{\"enable_thinking\": false}'")),
    ]
    for name, kw in spec:
        dest = name.replace("-", "_")
        if dest in skip:
            continue
        g.add_argument(f"--{name}", **kw)


def provider_of(a: Any) -> str:
    p = str(getattr(a, "provider", "") or "")
    return "" if p in ("default", "") else p


def extra_body_of(a: Any) -> Optional[dict]:
    raw = str(getattr(a, "extra_body", "") or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise SystemExit(f"--extra-body is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise SystemExit("--extra-body must be a JSON object")
    return data


def endpoint_config(a: Any, environ: Optional[dict] = None):
    from hexis.llm.env import llm_config
    return llm_config(environ=environ, profile=provider_of(a), model=str(getattr(a, "model", "") or ""),
                      base_url=str(getattr(a, "base_url", "") or ""),
                      api_key_env=str(getattr(a, "api_key_env", "") or ""))


def endpoint_record(a: Any, cfg: Any) -> dict:
    """The endpoint as recorded in build manifests: provider, model, base URL without credentials or query."""
    from hexis.llm.env import endpoint_record as _rec
    return _rec(cfg, provider_of(a))


@contextmanager
def open_model(a: Any, *, repair_chars: Optional[int] = 6000, environ: Optional[dict] = None,
               transport: Any = None) -> Iterator[tuple[Any, dict]]:
    """Open a client and a :class:`~hexis.llm.llm_client.ModelAdapter` for the parsed options ``a``."""
    from hexis.llm.llm_client import ModelAdapter, OpenAIClient
    cfg = endpoint_config(a, environ)
    max_tokens = int(getattr(a, "max_tokens", 16384))
    client = OpenAIClient(cfg.model, cfg.base_url, cfg.api_key, timeout=float(getattr(a, "llm_timeout", 180.0)),
                          max_retries=int(getattr(a, "llm_retries", 6)), transport=transport)
    client.max_tokens_ceiling = max(int(getattr(client, "max_tokens_ceiling", 0)), max_tokens)
    client.stream = bool(getattr(a, "stream", False))
    try:
        adapter = ModelAdapter(client, temperature=float(getattr(a, "temperature", 0.0)), max_tokens=max_tokens,
                               extra_body=extra_body_of(a), repair_chars=repair_chars)
        yield adapter, endpoint_record(a, cfg)
    finally:
        client.close()


__all__ = ["add_endpoint_args", "endpoint_config", "endpoint_record", "extra_body_of", "open_model", "provider_of"]

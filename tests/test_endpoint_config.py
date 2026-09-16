"""Model endpoint selection: flags override profiles and the environment; keys never leave the environment."""
from __future__ import annotations

import argparse
import json

import httpx
import pytest

from hexis.cli import _model
from hexis.llm.env import EnvError, endpoint_record, llm_config


def test_plain_environment():
    cfg = llm_config(environ={"MODEL": "m", "BASE_URL": "https://h/v1", "API_KEY": "sk-abcdef"})
    assert (cfg.model, cfg.base_url, cfg.api_key) == ("m", "https://h/v1", "sk-abcdef")


def test_flags_override_the_environment():
    cfg = llm_config(environ={"MODEL": "m", "BASE_URL": "https://h/v1", "API_KEY": "k"}, model="flag-model",
                     base_url="https://other/v1")
    assert (cfg.model, cfg.base_url) == ("flag-model", "https://other/v1")


def test_flags_fill_missing_settings():
    cfg = llm_config(environ={"API_KEY": "k"}, model="m", base_url="https://h/v1")
    assert cfg.model == "m" and cfg.api_key == "k"


def test_api_key_env_names_the_variable():
    cfg = llm_config(environ={"MY_KEY": "secret-1", "API_KEY": "other"}, model="m", base_url="u", api_key_env="MY_KEY")
    assert cfg.api_key == "secret-1"


def test_profile_and_flags():
    env = {"DEEPSEEK_API_KEY": "k", "DEEPSEEK_MODEL": "profile-model"}
    assert llm_config(environ=env, profile="deepseek").model == "profile-model"
    assert llm_config(environ=env, profile="deepseek", model="flag").model == "flag"


@pytest.mark.parametrize("env, kw, missing", [
    ({}, {}, ["MODEL", "BASE_URL", "API_KEY"]),
    ({"API_KEY": "sk-secret-value"}, {}, ["MODEL", "BASE_URL"]),
    ({"MODEL": "m", "BASE_URL": "u", "API_KEY": "sk-secret-value"}, {"api_key_env": "OTHER"}, ["OTHER"]),
])
def test_errors_name_only_the_missing_settings(env, kw, missing):
    with pytest.raises(EnvError) as exc:
        llm_config(environ=env, **kw)
    text = str(exc.value)
    assert all(name in text for name in missing)
    assert "sk-secret-value" not in text


def test_endpoint_record_has_no_credentials():
    cfg = llm_config(environ={"MODEL": "m", "BASE_URL": "https://user:pass@host:8443/v1?token=x", "API_KEY": "sk-9"})
    rec = endpoint_record(cfg, "")
    assert rec == {"provider": "default", "model": "m", "base_url": "https://host:8443/v1"}
    assert "sk-9" not in json.dumps(rec)


def test_open_model_sends_the_configured_request(monkeypatch):
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers.get("authorization"), json.loads(request.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"x": 1}'}, "finish_reason": "stop"}]})

    ap = argparse.ArgumentParser()
    _model.add_endpoint_args(ap)
    a = ap.parse_args(["--model", "my-model", "--base-url", "https://api.example/v1", "--api-key-env", "EXAMPLE_KEY",
                       "--temperature", "0.3", "--extra-body", '{"enable_thinking": false}'])
    env = {"EXAMPLE_KEY": "sk-example-1234"}
    with _model.open_model(a, environ=env, transport=httpx.MockTransport(handler)) as (model, rec):
        assert model.generate(prompt="p", values={}) == {"x": 1}
    url, auth, body = seen[0]
    assert url == "https://api.example/v1/chat/completions"
    assert auth == "Bearer sk-example-1234"
    assert body["model"] == "my-model" and body["temperature"] == 0.3 and body["enable_thinking"] is False
    assert rec == {"provider": "default", "model": "my-model", "base_url": "https://api.example/v1"}


def test_extra_body_must_be_a_json_object():
    ap = argparse.ArgumentParser()
    _model.add_endpoint_args(ap)
    with pytest.raises(SystemExit):
        _model.extra_body_of(ap.parse_args(["--extra-body", "[1, 2]"]))

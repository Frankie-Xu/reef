"""``reef serve`` reports every resolved setting with its source, masking credentials (issue #414)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from reef.service.deploy import orchestrator
from reef.service.deploy.diagnostics import mask_secrets, startup_report
from reef.service.deploy.inference import command_line_config
from reef.service.deploy.orchestrator import resolve_deployment_config


@pytest.fixture
def idle_stack(monkeypatch):
    class Stack:
        exit_code = 0

        def __init__(self, config, services, run_dir, ready_timeout_default, config_path, source_root=None):
            pass

        def start(self):
            pass

        def block(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    monkeypatch.setattr(orchestrator, "resolve_model_paths", lambda config: False)
    for key in list(os.environ):
        if key.startswith("REEF_"):
            monkeypatch.delenv(key)


def _report(raw, overrides, path):
    resolved, _ = resolve_deployment_config(raw, overrides, path)
    return startup_report(raw, resolved, overrides or {}, environ={}, from_file=True)


def test_file_command_line_automatic_and_default_sources(tmp_path):
    raw = {
        "schema-version": 2,
        "reef": {"port": 8000},
        "inference": {"model-path": "/models/demo", "options": {"mem-fraction-static": 0.8}, "upstream-api-key": None},
    }
    overrides = {"reef.port": "9000", "inference.options.mem-fraction-static": "0.6"}
    lines = _report(raw, overrides, tmp_path / "stack.yaml")
    assert lines[0] == "resolved settings:"
    settings = {line.split(" = ")[0].strip(): line.split(" = ")[1] for line in lines[1:]}
    assert settings["reef.port"] == "9000  (command line)"
    assert settings["inference.model-path"] == "/models/demo  (file)"
    assert settings["inference.backend"] == "sglang  (automatic)"
    assert settings["inference.options"] == '{"mem-fraction-static": "0.6"}  (file, command line)'
    assert settings["inference.upstream-api-key"] == "null  (file)"
    assert settings["reef.host"] == "127.0.0.1  (default)"
    assert settings["recipe.implementation"] == "recipe  (default)"
    assert "reef.console-origins" not in settings


def test_command_line_startup_reports_environment_and_flag_sources(monkeypatch):
    environ = {"REEF_UPSTREAM_URL": "http://localhost:8000", "REEF_TOKEN": "env-secret"}
    base = command_line_config(environ)
    resolved, _ = resolve_deployment_config(base, {"upstream-model": "demo"}, "<cli>", standard=True)
    lines = startup_report(base, resolved, {"upstream-model": "demo"}, environ=environ, from_file=False)
    assert "  inference.upstream-url = http://localhost:8000  (environment)" in lines
    assert "  inference.upstream-model = demo  (command line)" in lines
    assert "  reef.token = ****  (environment)" in lines
    assert "env-secret" not in "\n".join(lines)


def test_declared_environment_fallbacks_are_reported(tmp_path):
    raw = {
        "schema-version": 2,
        "recipe": {"implementation": "recipes.sao.recipe:SAORecipe"},
        "inference": {"model-path": "/models/demo"},
    }
    resolved, _ = resolve_deployment_config(raw, None, tmp_path / "stack.yaml")
    lines = startup_report(raw, resolved, {}, environ={"REEF_SAO_BATCH_SIZE": "4"}, from_file=True)
    assert "  recipe.config.batch-size = 4  (environment REEF_SAO_BATCH_SIZE)" in lines
    assert "  training.backend = slime  (automatic)" in lines


def test_secrets_are_masked_by_key_name_at_any_depth():
    assert mask_secrets(["a", "b"], "tokens") == ["****", "****"]
    assert mask_secrets("postgres://u:p@h/db", "record_database_url") == "****"
    assert mask_secrets(None, "upstream_api_key") is None
    assert mask_secrets({"lr": 1, "api-key": "k", "nested": {"password": "p"}}, "options") == {
        "lr": 1,
        "api-key": "****",
        "nested": {"password": "****"},
    }


@pytest.mark.usefixtures("idle_stack")
def test_serve_logs_the_config_source_and_masked_settings(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "stack.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema-version": 2,
                "reef": {"tokens": ["file-secret"]},
                "inference": {"upstream-url": "http://localhost:8000", "upstream-model": "demo"},
            }
        )
    )
    assert orchestrator._run_orchestrator(str(path), {"reef.port": "9000"}) == 0
    err = capsys.readouterr().err
    assert f"[reef] config: {path} (schema-version 2)" in err
    assert "[reef]   reef.port = 9000  (command line)" in err
    assert "[reef]   inference.upstream-model = demo  (file)" in err
    assert '[reef]   reef.tokens = ["****"]  (file)' in err
    assert "file-secret" not in err


@pytest.mark.usefixtures("idle_stack")
def test_serve_names_a_profile_and_an_unversioned_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        yaml.safe_dump(
            {
                "reef": {"recipe": "recipe", "upstream_url": "http://x", "upstream_model": "m"},
                "services": [{"name": "reef", "command": ["${REEF_PYTHON}", "-m", "reef.service"]}],
            }
        )
    )
    assert orchestrator._run_orchestrator(str(legacy), None) == 0
    err = capsys.readouterr().err
    assert f"[reef] config: {legacy} (unversioned layout)" in err
    assert "[reef]   inference.upstream-model = m  (file)" in err
    profile = Path(orchestrator.PROFILES_DIR) / "demo-profile.yaml"
    assert (
        orchestrator._config_source(profile, True) == f"config: profile demo-profile at {profile} (schema-version 2)"
    )

"""Tests for the `dio --version` flag and `dio check-config` command (issue #30)."""

from typer.testing import CliRunner

from dio import __version__
from dio.cli import app

runner = CliRunner()

VALID_YAML = """\
backends:
  - id: local-gpu
    url: http://localhost:8000
    tier: large
    models: [meta-llama/Llama-3.2-3B-Instruct]

scheduler:
  strategy: nlms
"""


def test_version_flag_prints_package_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_flag_matches_version_command():
    flag = runner.invoke(app, ["--version"])
    command = runner.invoke(app, ["version"])
    assert command.exit_code == 0
    assert flag.output.strip() == command.output.strip()


def _flat(result):
    """Output with Rich's line wrapping collapsed, so asserts do not depend on terminal width."""
    return " ".join(result.output.split())


def test_check_config_valid_file(tmp_path):
    cfg = tmp_path / "dio.yaml"
    cfg.write_text(VALID_YAML, encoding="utf-8")
    result = runner.invoke(app, ["check-config", str(cfg)])
    assert result.exit_code == 0
    assert "Config OK" in _flat(result)
    assert "1 backend(s)" in _flat(result)


def test_check_config_missing_file(tmp_path):
    result = runner.invoke(app, ["check-config", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
    assert "not found" in _flat(result).lower()


def test_check_config_unusable_config(tmp_path):
    cfg = tmp_path / "dio.yaml"
    cfg.write_text("backends: just-a-string\n", encoding="utf-8")
    result = runner.invoke(app, ["check-config", str(cfg)])
    assert result.exit_code != 0
    assert "Invalid config" in _flat(result)


def test_check_config_top_level_not_a_mapping(tmp_path):
    cfg = tmp_path / "dio.yaml"
    cfg.write_text("- one\n- two\n", encoding="utf-8")
    result = runner.invoke(app, ["check-config", str(cfg)])
    assert result.exit_code != 0
    assert "Invalid config" in _flat(result)


def test_check_config_malformed_yaml(tmp_path):
    cfg = tmp_path / "dio.yaml"
    cfg.write_text("backends: [unclosed\n  bad indent\n", encoding="utf-8")
    result = runner.invoke(app, ["check-config", str(cfg)])
    assert result.exit_code != 0
    assert result.output.strip() != ""

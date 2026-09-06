from __future__ import annotations

import json
import sys

import pytest

from customer_segmentation import cli


def test_cli_smoke_uses_bundled_config(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "run_pipeline", lambda output, config: {"mode": "smoke"})
    monkeypatch.setattr(sys, "argv", ["customer-segmentation", "smoke"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"mode": "smoke"}


def test_cli_reproduce_uses_requested_output(monkeypatch, capsys, tmp_path) -> None:
    captured = {}

    def fake_pipeline(output, config):
        captured["output"] = output
        return {"mode": "reproduce"}

    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        ["customer-segmentation", "reproduce", "--output-root", str(tmp_path)],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"mode": "reproduce"}
    assert captured["output"] == tmp_path


def test_cli_segment_passes_raw_override(monkeypatch, capsys, tmp_path) -> None:
    captured = {}

    def fake_segment(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {"mode": "segment"}

    monkeypatch.setattr(cli, "run_external_segmentation", fake_segment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "customer-segmentation",
            "segment",
            "--input",
            str(tmp_path / "input.csv"),
            "--output-dir",
            str(tmp_path / "output"),
            "--allow-raw-identifiers",
        ],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"mode": "segment"}
    assert captured["kwargs"]["identifier_policy"] == "raw"


def test_cli_fit_passes_pseudonymization_secret(monkeypatch, capsys, tmp_path) -> None:
    captured = {}

    def fake_fit(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"mode": "fit"}

    monkeypatch.setattr(cli, "fit_external_segmentation", fake_fit)
    monkeypatch.setenv("CUSTOMER_SEGMENTATION_ID_SALT", "test-only-salt-12345")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "customer-segmentation",
            "fit",
            "--input",
            str(tmp_path / "input.csv"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"mode": "fit"}
    assert captured["kwargs"] == {
        "identifier_policy": "pseudonymized",
        "identifier_salt": "test-only-salt-12345",
    }


def test_cli_score_raw_override_and_fail_on_review(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_score(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {"monitoring_gate_passed": False}

    monkeypatch.setattr(cli, "score_external_segmentation", fake_score)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "customer-segmentation",
            "score",
            "--input",
            str(tmp_path / "input.csv"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
            "--allow-raw-identifiers",
            "--fail-on-review",
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert captured["kwargs"]["identifier_policy"] == "raw"

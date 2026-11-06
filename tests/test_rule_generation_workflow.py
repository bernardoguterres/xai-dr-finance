"""Regressions for actual rule exports and the synthetic CLI workflow."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rule_generator import RuleGenerator
from threshold_discretizer import ThresholdDiscretizer


@pytest.mark.parametrize("format", ["json", "csv"])
def test_export_default_is_local_and_explicit_path_is_preserved(tmp_path, monkeypatch, format):
    monkeypatch.chdir(tmp_path)
    scores = np.array([[-2.0], [0.0], [2.0]])
    generator = RuleGenerator(min_samples_per_category=1)
    result = generator.generate_rules(scores, scores[:, :, None], ["feature"])

    generator.export_rules(result, format=format)
    default_file = tmp_path / "outputs" / f"rules_quantile.{format}"
    assert default_file.is_file()

    explicit_file = tmp_path / "custom" / f"chosen.{format}"
    generator.export_rules(result, format=format, output_path=explicit_file)
    assert explicit_file.read_text() == default_file.read_text()


def test_integer_domain_thresholds_change_actual_categories():
    scores = np.array([[-0.75], [0.0], [0.75]])
    labels, thresholds = ThresholdDiscretizer(
        method="domain", custom_thresholds={0: [-1.0, 1.0]}
    ).fit_transform(scores)
    assert thresholds == {0: [-1.0, 1.0]}
    assert labels[:, 0].tolist() == [1, 1, 1]


def test_demo_runs_without_pythonpath_and_exports_consistent_rules(tmp_path):
    """Use real PCA, XGBoost and SHAP; catch demo wiring, not just API behavior."""
    script = Path(__file__).resolve().parents[1] / "demos" / "demo_rule_generation.py"
    destination = tmp_path / "nested" / "artifacts"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg",
        MPLCONFIGDIR=str(tmp_path / "matplotlib"),
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(destination)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (tmp_path / "outputs").exists()

    with (destination / "all_rules.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    readable = (destination / "rules_human_readable.txt").read_text()
    for strategy in ["quantile", "std", "domain"]:
        exported = json.loads((destination / f"rules_{strategy}.json").read_text())
        strategy_rows = [row for row in rows if row["Strategy"] == strategy]
        assert exported["n_rules"] == len(exported["rules"]) == len(strategy_rows)
        assert exported["n_rules"] > 0
        for rule, row in zip(exported["rules"], strategy_rows):
            assert rule["rule_text"] == row["Rule_Text"]
            assert rule["rule_text"] in readable
            assert rule["factor"] == row["Factor"]
            if strategy == "domain":
                lower, upper = rule["threshold_bounds"]
                if rule["category"] == "LOW":
                    assert upper == -1.0
                elif rule["category"] == "MEDIUM":
                    assert [lower, upper] == [-1.0, 1.0]
                else:
                    assert rule["category"] == "HIGH"
                    assert lower == 1.0

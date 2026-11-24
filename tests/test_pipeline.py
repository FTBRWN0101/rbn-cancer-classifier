"""Run the whole pipeline on a small made-up dataset, as `python src` runs it."""

import subprocess
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"


def write_synthetic_arff(
    path: Path, n_cancer: int = 40, n_normal: int = 24, n_features: int = 40
) -> None:
    """Write an ARFF in the dataset's layout, with a class shift on most features."""
    rng = np.random.default_rng(0)
    labels = ["Cancer"] * n_cancer + ["Normal"] * n_normal
    shift = np.where(np.arange(n_features) < 35, 0.3, 0.0)
    rows = [
        rng.uniform(0, 0.7, n_features) + (shift if label == "Cancer" else 0)
        for label in labels
    ]
    lines = ["@relation synthetic", ""]
    lines += [f"@attribute MZ{i} numeric" for i in range(n_features)]
    lines += ["@attribute Class {Cancer,Normal}", "", "@data", ""]
    lines += [
        ",".join(f"{v:.6f}" for v in row) + f",{label}"
        for row, label in zip(rows, labels, strict=True)
    ]
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_pipeline_runs_end_to_end(tmp_path):
    write_synthetic_arff(tmp_path / "data" / "raw_data" / "Ovarian.arff")
    # Output goes to a pipe, which on Windows also checks the UTF-8 switch in src/__main__.py.
    result = subprocess.run(
        [sys.executable, str(SRC)],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    processed = tmp_path / "data" / "processed"
    assert np.load(processed / "train_labels.npy").tolist().count(0) == 28
    assert np.load(processed / "features" / "selected_features.npy").shape == (30,)
    metrics = tmp_path / "results" / "metrics"
    for name in ("cm_rbn", "cm_random_forest"):
        assert np.load(metrics / f"{name}.npy").sum() == 20
    for name in ("auc_ROC_rbn", "auc_ROC_rf"):
        assert 0.5 < np.load(metrics / f"{name}.npy")[0] <= 1.0
    features = np.load(
        metrics / "feature_importance_Feature Importance RBN.npz", allow_pickle=True
    )["features"]
    assert features.tolist()[:2] == ["attractor_length", "mean_activity"]
    assert (tmp_path / "results" / "figures" / "confusion_matrix_rbn.png").exists()

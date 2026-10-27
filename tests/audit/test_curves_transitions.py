import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import needs_deposit
from srwc import paths

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(script, *args):
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), *args], check=True, env=env)


@needs_deposit
def test_curves_equal_deposit(tmp_path):
    out = tmp_path / "curves.csv"
    _run("04_curves.py", "--clusterings", str(paths.deposit("clusterings_fullscan")), "--out", str(out))
    ours = pd.read_csv(out).sort_values(["Source", "K"]).reset_index(drop=True)
    ref = pd.read_csv(paths.deposit("curves", "purity_metrics_extended.csv")).sort_values(["Source", "K"]).reset_index(drop=True)
    assert list(ours.columns) == list(ref.columns)
    pd.testing.assert_frame_equal(ours, ref, check_exact=False, rtol=1e-12, atol=1e-12)


@needs_deposit
@pytest.mark.parametrize("name", ["nva_k3_gamma0.2_kmeans.csv", "extended_k120_gamma0.2_kmeans.csv"])
def test_transitions_equal_deposit(tmp_path, name):
    _run("05_transitions.py", "--out", str(tmp_path))
    ours = pd.read_csv(tmp_path / name)
    ref = pd.read_csv(paths.deposit("transitions", name))
    assert ours.columns.tolist() == ref.columns.tolist()
    assert ours.source_cluster.tolist() == ref.source_cluster.tolist()
    assert ours.majority_pos.tolist() == ref.majority_pos.tolist()
    assert ours.n_tokens.tolist() == ref.n_tokens.tolist()
    np.testing.assert_allclose(ours.iloc[:, 3:].to_numpy(), ref.iloc[:, 3:].to_numpy(), rtol=0, atol=1e-6)

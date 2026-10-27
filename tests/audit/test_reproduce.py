"""The deposit's own check script: 62 reported values recomputed from the deposited files."""
import subprocess
import sys

from conftest import needs_deposit
from srwc import paths


@needs_deposit
def test_deposit_reproduce_script_passes():
    run = subprocess.run([sys.executable, str(paths.deposit("reproduce.py"))], cwd=paths.deposit(),
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]

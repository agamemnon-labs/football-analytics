import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_priors_table_matches_code():
    result = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "priors_table.py"),
                             "--check"])
    assert result.returncode == 0, "docs/PRIORS.md is out of date: run scripts/priors_table.py"

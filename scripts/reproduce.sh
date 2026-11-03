#!/usr/bin/env bash
# Regenerate evaluation/results/Agamemnon.csv from a clean clone plus the match bundle in data/.
# CPU only; about a minute. Prints the md5 so two runs can be compared.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -q -r requirements.txt
python -m pip install -q -e . --no-deps
python -m pytest -q
python -m agamemnon.pipeline --out evaluation/results/Agamemnon.csv
python evaluation/validate_submission.py evaluation/results/Agamemnon.csv
python - <<'PY'
import hashlib
print("md5", hashlib.md5(open("evaluation/results/Agamemnon.csv", "rb").read()).hexdigest())
PY

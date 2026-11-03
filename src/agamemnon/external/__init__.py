"""Converters and benchmark tooling for external, openly licensed matches."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXTERNAL_DIR = os.environ.get("AGAMEMNON_EXTERNAL_DIR") or os.path.join(ROOT, "external")
CACHE_DIR = os.path.join(EXTERNAL_DIR, "cache")

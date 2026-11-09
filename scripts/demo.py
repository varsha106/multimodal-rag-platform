#!/usr/bin/env python3
"""Run the package CLI from a source checkout without installing it first."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

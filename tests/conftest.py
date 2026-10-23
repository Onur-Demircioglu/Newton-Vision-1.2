from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

for core_path in (PROJECT_ROOT / "build").glob("*/_core*.pyd"):
    sys.path.insert(0, str(core_path.parent))
    break

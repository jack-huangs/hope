from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path) -> None:
    """Load simple KEY=value lines without printing or overwriting process env."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("Invalid environment file line")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.isidentifier():
            raise ValueError("Invalid environment variable name")
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """Load environment variables from .env."""
    load_dotenv(PROJECT_ROOT / ".env")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file relative to project root or by absolute path."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")

    return data


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return Path."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.mkdir(parents=True, exist_ok=True)
    return path
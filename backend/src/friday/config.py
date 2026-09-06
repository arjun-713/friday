"""Load non-secret runtime configuration from YAML and secrets from the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .paths import project_root


def load_runtime_config(path: Path | None = None) -> dict[str, Any]:
    """Load the YAML config and local dotenv file without allowing secrets in YAML."""

    root = project_root()
    env_path = Path(os.getenv("FRIDAY_ENV_FILE", ".env"))
    if not env_path.is_absolute() and not env_path.is_file() and (root / "backend" / env_path).is_file():
        env_path = root / "backend" / env_path
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    config_path = path or Path(os.getenv("FRIDAY_CONFIG", "config.yml"))
    if not config_path.is_absolute() and not config_path.is_file() and (root / "backend" / config_path).is_file():
        config_path = root / "backend" / config_path
    if not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section: Any = config
    for part in name.split("."):
        section = section.get(part, {}) if isinstance(section, dict) else {}
    return section if isinstance(section, dict) else {}

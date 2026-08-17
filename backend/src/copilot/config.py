"""Load non-secret runtime configuration from YAML and secrets from the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_runtime_config(path: Path | None = None) -> dict[str, Any]:
    """Load the YAML config and local dotenv file without allowing secrets in YAML."""

    env_path = Path(os.getenv("FRIDAY_ENV_FILE", ".env"))
    if not env_path.is_file() and (Path("backend") / env_path).is_file():
        env_path = Path("backend") / env_path
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    config_path = path or Path(os.getenv("FRIDAY_CONFIG", "config.yml"))
    if not config_path.is_file() and (Path("backend") / config_path).is_file():
        config_path = Path("backend") / config_path
    if not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section: Any = config
    for part in name.split("."):
        section = section.get(part, {}) if isinstance(section, dict) else {}
    return section if isinstance(section, dict) else {}

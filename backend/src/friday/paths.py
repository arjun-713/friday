"""Single home for repository-relative path resolution.

Every CLI, runner, and benchmark used to hard-code ``Path("data/...")``,
which silently assumed the process working directory was the repository
root. The API server instead anchored paths at ``FRIDAY_ROOT``. This module
unifies both: paths resolve under ``$FRIDAY_ROOT`` when set (Docker sets it
to ``/app``), otherwise under the repository root derived from this file's
location. Run ``FRIDAY_ROOT`` must point at the checkout when the package is
imported from an installed location rather than the source tree.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root (``backend/src/friday/paths.py`` parents)."""

    configured = os.getenv("FRIDAY_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return project_root() / "data"


def manuals_dir() -> Path:
    return data_dir() / "manuals"


def raw_dir() -> Path:
    return data_dir() / "raw"


def cleaned_dir() -> Path:
    return data_dir() / "cleaned"


def chunks_dir() -> Path:
    return data_dir() / "chunks"


def index_dir() -> Path:
    return data_dir() / "index"


def models_dir() -> Path:
    return data_dir() / "models"


def assets_dir() -> Path:
    return data_dir() / "assets"


def images_dir() -> Path:
    return assets_dir() / "images"


def config_dir() -> Path:
    return project_root() / "config"


def source_registry() -> Path:
    """Tracked 21-manual corpus manifest (project input)."""

    return config_dir() / "source_registry.json"


def resolved_registry() -> Path:
    """Generated per-file hashes written by the metadata registry runner."""

    return raw_dir() / "source_registry.json"


def eval_dir() -> Path:
    return project_root() / "eval"


def tmp_dir() -> Path:
    return project_root() / "tmp"


def repo_relative(path: Path) -> str:
    """Render *path* as a repo-root-relative posix string for stored artifacts.

    Generated JSON (parse output, registries, reports) historically records
    ``data/...`` relative paths. Keeping that format makes artifacts portable
    across machines instead of embedding absolute checkout locations.
    """

    path = Path(path)
    try:
        return path.relative_to(project_root()).as_posix()
    except ValueError:
        return path.as_posix()

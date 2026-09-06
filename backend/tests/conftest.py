"""Make repo-root packages importable when pytest runs from ``backend/``.

CI executes ``python -m pytest`` with ``working-directory: backend``. The
backend suite also covers ``eval/`` and ``scripts/`` helpers, which live at
the repository root, so the root is added to ``sys.path`` here. The ``friday``
package itself resolves via ``pythonpath = ["src"]`` in ``pyproject.toml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

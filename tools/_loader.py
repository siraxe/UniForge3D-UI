"""Hot-reload loader for the geometry tools under ``tools/``.

Every tool is a plain package (``tools/reproject``, ``tools/polish``,
``tools/remesh``) whose sources live on disk as normal ``.py`` files.
``load_tool`` drops any cached copy from ``sys.modules`` and re-imports the
package, so edits made while the app is running take effect on the next
job - no restart needed.

The UI only ever calls ``load_tool`` from inside its worker threads, so a
job that is already running keeps the byte-code it started with and is
never swapped mid-flight.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

__all__ = ["load_tool"]


def load_tool(name: str) -> ModuleType:
    """Import ``tools.<name>`` fresh from disk (ignores any cached copy)."""
    package = f"tools.{name}"
    for module in [m for m in list(sys.modules)
                   if m == package or m.startswith(package + ".")]:
        del sys.modules[module]
    return importlib.import_module(package)

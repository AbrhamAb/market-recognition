"""Type stubs for `sklearn` package (minimal, workspace-local).
This folder provides the modules used by the project so the language
server stops reporting missing-module errors.
"""

from typing import Any

# Expose the `metrics` submodule for static analysis
metrics: Any

__all__ = ["metrics"]

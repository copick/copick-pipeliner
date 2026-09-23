"""Where the external programs are.

Mirrors pipeliner's own ``PIPELINER_CTFFIND_EXECUTABLE`` convention: an environment
variable names the executable of a tool that lives outside the control process's
environment; when unset, ``PATH`` is searched; when that fails too, the bare name is
returned so the failure is a clear "command not found" in the job's ``run.err`` rather
than an exception at class-load time (pipeliner loads every job class when it lists job
types, and a missing optional tool must not make the whole registry unavailable).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ENV_COPICK = "PIPELINER_COPICK_EXECUTABLE"
ENV_OCTOPI = "PIPELINER_OCTOPI_EXECUTABLE"
ENV_TOOLS = "PIPELINER_COPICK_PIPELINER_TOOLS_EXECUTABLE"

TOOLS_NAME = "copick-pipeliner-tools"


def _resolve(env_var: str, name: str) -> str:
    configured = (os.environ.get(env_var) or "").strip()
    if configured:
        return configured
    found = shutil.which(name)
    return found or name


def copick_exe() -> str:
    """The ``copick`` CLI (with the copick-utils / copick-easymode / copick-torch plugins)."""
    return _resolve(ENV_COPICK, "copick")


def octopi_exe() -> str:
    """The ``octopi`` CLI."""
    return _resolve(ENV_OCTOPI, "octopi")


def tools_exe() -> str:
    """``copick-pipeliner-tools``: this package's own CLI, which must run where copick is.

    Resolution order: the explicit variable, a sibling of the copick executable (the
    picking environment's ``bin/``), ``PATH``, the bare name.
    """
    configured = (os.environ.get(ENV_TOOLS) or "").strip()
    if configured:
        return configured
    copick = copick_exe()
    if os.sep in copick:
        sibling = Path(copick).parent / TOOLS_NAME
        if sibling.is_file():
            return str(sibling)
    return shutil.which(TOOLS_NAME) or TOOLS_NAME

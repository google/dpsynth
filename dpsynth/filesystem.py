# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filesystem abstraction for checkpoint I/O.

Provides a simple ``FileSystem`` dataclass that wraps a handful of filesystem
operations (open, exists, makedirs, remove) so that library code can checkpoint
intermediate state without hardcoding any specific storage backend.

Local filesystem behavior is used by default. To read and write on a remote
or networked filesystem, construct a ``FileSystem`` with callables from the
appropriate storage client::

    fs = FileSystem(
        open=client.open,
        exists=client.exists,
        makedirs=client.makedirs,
        remove=client.remove,
    )
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
import dataclasses
import os
from typing import Any, IO


@dataclasses.dataclass(frozen=True)
class FileSystem:
  """Pluggable filesystem for checkpoint I/O.

  Wraps the operations that the library needs for checkpointing:

  - ``open(path, mode) -> file``: Open a file for reading or writing.
  - ``exists(path) -> bool``: Check whether a path exists.
  - ``makedirs(path) -> None``: Create a directory (and parents).
  - ``remove(path) -> None``: Delete a file.

  The defaults use Python's built-in ``open`` and the ``os`` module, so
  local-filesystem checkpointing works out of the box with no extra arguments.
  To use a remote or networked filesystem, construct a ``FileSystem`` with
  the appropriate callables.

  The library does **not** handle encryption or access control. When saving
  sensitive data (e.g., exact marginals), it is the caller's responsibility to
  ensure that the directory has appropriate protections.

  Attributes:
    open: Callable that opens a file given (path, mode).
    exists: Callable that checks whether a path exists.
    makedirs: Callable that creates a directory and its parents.
    remove: Callable that deletes a file.
  """

  open: Callable[..., IO[Any]] = builtins.open
  exists: Callable[[str], bool] = os.path.exists
  makedirs: Callable[..., None] = dataclasses.field(
      default_factory=lambda: lambda path: os.makedirs(path, exist_ok=True)
  )
  remove: Callable[[str], None] = os.remove

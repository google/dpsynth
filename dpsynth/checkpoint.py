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

"""Checkpointing utilities for long-running mechanism synthesis.

Provides :class:`Checkpointer`, which serializes and deserializes intermediate
mechanism state (e.g. exact marginals, noisy measurements, graphical models)
using :mod:`mbi` pytree serialization on top of :mod:`etils.epath`.
"""

from __future__ import annotations

import dataclasses
import io
from typing import Any

from etils import epath
import mbi


@dataclasses.dataclass(frozen=True)
class Checkpointer:
  """Saves and restores intermediate mechanism state as .npz checkpoints.

  When ``working_dir`` is None (the default), all save/load operations are
  no-ops, allowing callers to disable checkpointing without branching.
  When ``working_dir`` is provided, intermediate mechanism state is persisted
  directly under that directory as .npz files using ``mbi.save`` and
  ``mbi.load``.

  Attributes:
    working_dir: Base directory path for checkpoint files (supports local,
      Cloud, and remote paths via epath.Path). If None, checkpointing is
      disabled.
  """

  working_dir: epath.PathLike | None = None

  @property
  def path(self) -> epath.Path | None:
    """The resolved working directory path, or None if disabled."""
    return (
        epath.Path(self.working_dir) if self.working_dir is not None else None
    )

  def save(self, name: str, obj: Any) -> None:
    """Saves an object to the working directory (no-op if disabled).

    Args:
      name: Filename to write the object to (e.g. 'model.npz').
      obj: A JAX pytree to serialize (e.g. a CliqueVector, model, or list of
        measurements).
    """
    if self.path is None:
      return
    self.path.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    mbi.save(obj, buf)
    (self.path / name).write_bytes(buf.getvalue())

  def load(self, name: str) -> Any | None:
    """Loads an object from the working directory, or None if absent/disabled.

    Args:
      name: Filename of the checkpointed object.

    Returns:
      The deserialized object, or None if checkpointing is disabled or the
      file does not exist.
    """
    if self.path is None:
      return None
    target = self.path / name
    if not target.exists():
      return None
    return mbi.load(io.BytesIO(target.read_bytes()))

  def exists(self, name: str) -> bool:
    """Returns True if the named checkpoint file exists."""
    if self.path is None:
      return False
    return (self.path / name).exists()

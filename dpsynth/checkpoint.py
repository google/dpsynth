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

"""Checkpoint manager for saving and resuming intermediate mechanism state."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import io
import os
from typing import Any

from dpsynth import filesystem
import mbi


@dataclasses.dataclass
class Checkpointer:
  """Saves and restores intermediate mechanism state under a working directory.

  A ``Checkpointer`` lets a long-running mechanism persist intermediate results
  so that a preempted run can resume from the latest completed phase instead of
  restarting. Objects are serialized as ``.npz`` files via ``mbi.save`` /
  ``mbi.load``, which round-trip arbitrary JAX pytrees (e.g. ``CliqueVector``,
  ``MarkovRandomField``, and lists of ``LinearMeasurement``).

  State is organized into named subdirectories of ``working_dir``. Callers are
  expected to separate DP-safe outputs (e.g. noisy measurements, the estimated
  model) from sensitive intermediates (e.g. exact marginals) by writing them to
  different subdirectories. This class performs no encryption or access control:
  when saving sensitive data it is the caller's responsibility to ensure the
  directory has appropriate protections.

  When ``working_dir`` is ``None`` (the default) every operation is a no-op and
  ``load`` always returns ``None``, so mechanisms behave exactly as if
  checkpointing were disabled.

  Attributes:
    working_dir: Root directory for checkpoint files, or None to disable
      checkpointing entirely.
    fs: Filesystem abstraction used for all I/O. Defaults to the local
      filesystem; override for remote or networked storage.
  """

  working_dir: str | None = None
  fs: filesystem.FileSystem = dataclasses.field(
      default_factory=filesystem.FileSystem
  )

  def setup(self, *subdirs: str) -> None:
    """Creates the working directory and the given subdirectories.

    Args:
      *subdirs: Names of subdirectories to create under ``working_dir``.
    """
    if self.working_dir is None:
      return
    for subdir in subdirs:
      self.fs.makedirs(os.path.join(self.working_dir, subdir))

  def load(self, subdir: str, name: str) -> Any | None:
    """Loads a checkpointed object, returning None if it does not exist.

    Args:
      subdir: Subdirectory of ``working_dir`` containing the file.
      name: Filename of the checkpointed object.

    Returns:
      The deserialized object, or None if checkpointing is disabled or the
      file does not exist.
    """
    if self.working_dir is None:
      return None
    path = os.path.join(self.working_dir, subdir, name)
    if not self.fs.exists(path):
      return None
    with self.fs.open(path, 'rb') as f:
      return mbi.load(io.BytesIO(f.read()))

  def save(self, subdir: str, name: str, obj: Any) -> None:
    """Saves an object to ``<working_dir>/<subdir>/<name>``.

    Args:
      subdir: Subdirectory of ``working_dir`` to write to.
      name: Filename to write the object to.
      obj: A JAX pytree to serialize (e.g. a CliqueVector, model, or list of
        measurements).
    """
    if self.working_dir is None:
      return
    buf = io.BytesIO()
    mbi.save(obj, buf)
    path = os.path.join(self.working_dir, subdir, name)
    with self.fs.open(path, 'wb') as f:
      f.write(buf.getvalue())

  def cleanup(self, subdir: str, names: Sequence[str]) -> None:
    """Deletes the named files from a subdirectory, ignoring missing ones.

    Args:
      subdir: Subdirectory of ``working_dir`` to delete files from.
      names: Filenames to delete. Files that do not exist are skipped.
    """
    if self.working_dir is None:
      return
    for name in names:
      path = os.path.join(self.working_dir, subdir, name)
      if self.fs.exists(path):
        self.fs.remove(path)

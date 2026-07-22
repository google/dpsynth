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

"""Checkpoint manager for saving and resuming intermediate mechanism state.

Storage is separated into two roles so that resume-only state is never confused
with inspectable output:

- A :class:`PrivateStore` holds resume-only state (e.g. exact marginals, noisy
  measurements, the estimated model). It is read back to resume a preempted run
  and is never intended for human inspection. It may contain sensitive
  intermediates, so it must live somewhere with appropriate protections.
- A :class:`PublicSink` holds DP-safe outputs that are meant to be inspected or
  consumed downstream. It is write-only (egress); the library never reads it
  back.

The default local implementations (:class:`LocalDirStore`,
:class:`LocalDirSink`)
are backed by the :mod:`dpsynth.filesystem` abstraction. The same two roles map
cleanly onto a Trusted Execution Environment's release APIs: a private store
onto the recovery-info channel, and a public sink onto the unencrypted-release
channel.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import io
import os
from typing import Any, Protocol, cast

from dpsynth import filesystem
import mbi


class PrivateStore(Protocol):
  """Resume-only, opaque checkpoint storage within the trust boundary.

  Holds internal state needed to resume a preempted run. Contents are never
  meant for human inspection and may include sensitive intermediates (e.g.
  exact marginals). In a TEE this maps to the recovery-info channel.
  """

  def put(self, name: str, data: bytes) -> None:
    ...

  def get(self, name: str) -> bytes | None:
    ...

  def delete(self, names: Sequence[str]) -> None:
    ...


class PublicSink(Protocol):
  """DP-safe, inspectable output storage (write-only egress).

  Holds outputs intended for downstream consumption or human inspection. In a
  TEE this maps to the unencrypted-release channel, which cannot be read back.
  """

  def export(self, name: str, data: bytes) -> None:
    ...


@dataclasses.dataclass(frozen=True)
class LocalDirStore:
  """A :class:`PrivateStore` backed by a filesystem directory.

  Attributes:
    root: Directory under which checkpoint files are written.
    fs: Filesystem abstraction for all I/O. Defaults to the local filesystem.
  """

  root: str
  fs: filesystem.FileSystem = dataclasses.field(
      default_factory=filesystem.FileSystem
  )

  def put(self, name: str, data: bytes) -> None:
    self.fs.makedirs(self.root)
    with self.fs.open(os.path.join(self.root, name), 'wb') as f:
      f.write(data)

  def get(self, name: str) -> bytes | None:
    path = os.path.join(self.root, name)
    if not self.fs.exists(path):
      return None
    with self.fs.open(path, 'rb') as f:
      return cast(bytes, f.read())

  def delete(self, names: Sequence[str]) -> None:
    for name in names:
      path = os.path.join(self.root, name)
      if self.fs.exists(path):
        self.fs.remove(path)


@dataclasses.dataclass(frozen=True)
class LocalDirSink:
  """A :class:`PublicSink` backed by a filesystem directory.

  Attributes:
    root: Directory under which exported files are written.
    fs: Filesystem abstraction for all I/O. Defaults to the local filesystem.
  """

  root: str
  fs: filesystem.FileSystem = dataclasses.field(
      default_factory=filesystem.FileSystem
  )

  def export(self, name: str, data: bytes) -> None:
    self.fs.makedirs(self.root)
    with self.fs.open(os.path.join(self.root, name), 'wb') as f:
      f.write(data)


@dataclasses.dataclass
class Checkpointer:
  """Saves resume state privately and exports DP-safe outputs publicly.

  A ``Checkpointer`` separates two storage roles:

  - ``private``: a :class:`PrivateStore` for resume-only state (e.g. exact
    marginals, noisy measurements, the estimated model). This data is used
    solely to resume a preempted run and is never meant for human inspection.
  - ``public``: a :class:`PublicSink` for DP-safe outputs that are intended to
    be inspected or consumed downstream.

  Either role may be ``None`` to disable it. When ``private`` is ``None`` the
  resume methods (:meth:`save`, :meth:`load`, :meth:`cleanup`) are no-ops and
  :meth:`load` returns ``None``, so mechanisms behave exactly as if
  checkpointing were disabled. When ``public`` is ``None`` :meth:`export` is a
  no-op.

  Objects are serialized as ``.npz`` blobs via ``mbi.save`` / ``mbi.load``,
  which round-trip arbitrary JAX pytrees (e.g. ``CliqueVector``,
  ``MarkovRandomField``, and lists of ``LinearMeasurement``).

  Attributes:
    private: Store for resume-only checkpoint state, or None to disable.
    public: Sink for DP-safe inspectable outputs, or None to disable.
  """

  private: PrivateStore | None = None
  public: PublicSink | None = None

  @classmethod
  def local(
      cls,
      working_dir: str,
      fs: filesystem.FileSystem | None = None,
  ) -> Checkpointer:
    """Returns a Checkpointer backed by ``private/`` and ``public/`` subdirs.

    Args:
      working_dir: Root directory under which ``private/`` and ``public/``
        subdirectories are created lazily on first write.
      fs: Filesystem abstraction for all I/O. Defaults to the local filesystem.
    """
    fs = fs if fs is not None else filesystem.FileSystem()
    return cls(
        private=LocalDirStore(os.path.join(working_dir, 'private'), fs),
        public=LocalDirSink(os.path.join(working_dir, 'public'), fs),
    )

  def load(self, name: str) -> Any | None:
    """Loads resume state, returning None if absent or private is disabled.

    Args:
      name: Filename of the checkpointed object.

    Returns:
      The deserialized object, or None if there is no private store or the
      object has not been checkpointed.
    """
    if self.private is None:
      return None
    data = self.private.get(name)
    if data is None:
      return None
    return mbi.load(io.BytesIO(data))

  def save(self, name: str, obj: Any) -> None:
    """Saves resume state to the private store (no-op if disabled).

    Args:
      name: Filename to write the object to.
      obj: A JAX pytree to serialize (e.g. a CliqueVector, model, or list of
        measurements).
    """
    if self.private is None:
      return
    buf = io.BytesIO()
    mbi.save(obj, buf)
    self.private.put(name, buf.getvalue())

  def cleanup(self, names: Sequence[str]) -> None:
    """Deletes named resume files from the private store, ignoring missing ones.

    Args:
      names: Filenames to delete. Files that do not exist are skipped.
    """
    if self.private is None:
      return
    self.private.delete(names)

  def export(self, name: str, obj: Any) -> None:
    """Exports a DP-safe object to the public sink (no-op if disabled).

    Args:
      name: Filename to export the object to.
      obj: A JAX pytree to serialize. Callers are responsible for ensuring that
        only DP-safe data is exported publicly.
    """
    if self.public is None:
      return
    buf = io.BytesIO()
    mbi.save(obj, buf)
    self.public.export(name, buf.getvalue())

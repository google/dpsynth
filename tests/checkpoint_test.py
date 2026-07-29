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

import os
import tempfile

from absl.testing import absltest
from dpsynth import checkpoint as checkpoint_lib
from dpsynth import filesystem
import jax.numpy as jnp
import numpy as np


class CheckpointerTest(absltest.TestCase):

  def test_disabled_by_default(self):
    """A Checkpointer with no stores is a no-op for every method."""
    ckpt = checkpoint_lib.Checkpointer()
    ckpt.save('x.npz', {'a': jnp.arange(3)})  # No-op.
    self.assertIsNone(ckpt.load('x.npz'))
    ckpt.export('x.npz', {'a': jnp.arange(3)})  # No-op.
    ckpt.cleanup(['x.npz'])  # Should not raise.

  def test_save_load_roundtrip(self):
    obj = {'a': jnp.arange(3), 'b': jnp.ones((2, 2))}
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt = checkpoint_lib.Checkpointer.local(tmpdir)
      self.assertIsNone(ckpt.load('x.npz'))
      ckpt.save('x.npz', obj)
      loaded = ckpt.load('x.npz')
      np.testing.assert_array_equal(loaded['a'], obj['a'])
      np.testing.assert_array_equal(loaded['b'], obj['b'])

  def test_local_uses_private_and_public_subdirs(self):
    """Resume state lands under private/ and exports land under public/."""
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt = checkpoint_lib.Checkpointer.local(tmpdir)
      ckpt.save('model.npz', {'a': jnp.arange(2)})
      ckpt.export('synthetic.npz', {'b': jnp.arange(2)})
      self.assertTrue(
          os.path.isfile(os.path.join(tmpdir, 'private', 'model.npz'))
      )
      self.assertTrue(
          os.path.isfile(os.path.join(tmpdir, 'public', 'synthetic.npz'))
      )

  def test_public_exports_are_not_visible_to_load(self):
    """load reads only resume state; it never reads back public exports."""
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt = checkpoint_lib.Checkpointer.local(tmpdir)
      ckpt.export('synthetic.npz', {'b': jnp.arange(2)})
      self.assertIsNone(ckpt.load('synthetic.npz'))

  def test_export_disabled_when_public_is_none(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt = checkpoint_lib.Checkpointer(
          private=checkpoint_lib.LocalDirStore(os.path.join(tmpdir, 'private'))
      )
      ckpt.export('x.npz', {'a': jnp.arange(1)})  # No-op, no public sink.
      self.assertFalse(os.path.isdir(os.path.join(tmpdir, 'public')))

  def test_cleanup_removes_files_and_tolerates_missing(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt = checkpoint_lib.Checkpointer.local(tmpdir)
      ckpt.save('marginals.npz', {'a': jnp.arange(2)})
      self.assertIsNotNone(ckpt.load('marginals.npz'))
      # Present and missing files can be requested together.
      ckpt.cleanup(['marginals.npz', 'missing.npz'])
      self.assertIsNone(ckpt.load('marginals.npz'))

  def test_routes_io_through_filesystem(self):
    """The local stores perform all I/O through the provided FileSystem."""
    events = []
    real = filesystem.FileSystem()

    def recording_open(path, mode):
      events.append(('open', mode))
      return real.open(path, mode)

    with tempfile.TemporaryDirectory() as tmpdir:
      fs = filesystem.FileSystem(
          open=recording_open,
          exists=real.exists,
          makedirs=real.makedirs,
          remove=real.remove,
      )
      ckpt = checkpoint_lib.Checkpointer.local(tmpdir, fs=fs)
      ckpt.save('x.npz', {'a': jnp.arange(1)})
      self.assertIsNotNone(ckpt.load('x.npz'))
      self.assertIn(('open', 'wb'), events)
      self.assertIn(('open', 'rb'), events)

  def test_accepts_custom_stores(self):
    """Custom in-memory stores can back the two roles (e.g. a TEE adapter)."""

    class MemStore:
      """A minimal PrivateStore backed by a dict."""

      def __init__(self):
        self.blobs = {}

      def put(self, name, data):
        self.blobs[name] = data

      def get(self, name):
        return self.blobs.get(name)

      def delete(self, names):
        for name in names:
          self.blobs.pop(name, None)

    class MemSink:
      """A minimal PublicSink backed by a dict."""

      def __init__(self):
        self.released = {}

      def export(self, name, data):
        self.released[name] = data

    store, sink = MemStore(), MemSink()
    ckpt = checkpoint_lib.Checkpointer(private=store, public=sink)
    ckpt.save('model.npz', {'a': jnp.arange(3)})
    ckpt.export('synthetic.npz', {'b': jnp.arange(3)})
    self.assertIn('model.npz', store.blobs)
    self.assertIn('synthetic.npz', sink.released)
    loaded = ckpt.load('model.npz')
    np.testing.assert_array_equal(loaded['a'], jnp.arange(3))


if __name__ == '__main__':
  absltest.main()

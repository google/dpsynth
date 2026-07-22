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
    """A Checkpointer with no working_dir is a no-op."""
    ckpt = checkpoint_lib.Checkpointer()
    ckpt.setup('public', 'sensitive')  # Should not raise.
    ckpt.save('public', 'x.npz', {'a': jnp.arange(3)})  # No-op.
    self.assertIsNone(ckpt.load('public', 'x.npz'))
    ckpt.cleanup('public', ['x.npz'])  # Should not raise.

  def test_setup_creates_subdirs(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt = checkpoint_lib.Checkpointer(working_dir=tmpdir)
      ckpt.setup('public', 'sensitive')
      self.assertTrue(os.path.isdir(os.path.join(tmpdir, 'public')))
      self.assertTrue(os.path.isdir(os.path.join(tmpdir, 'sensitive')))

  def test_save_load_roundtrip(self):
    obj = {'a': jnp.arange(3), 'b': jnp.ones((2, 2))}
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt = checkpoint_lib.Checkpointer(working_dir=tmpdir)
      ckpt.setup('public')
      self.assertIsNone(ckpt.load('public', 'x.npz'))
      ckpt.save('public', 'x.npz', obj)
      loaded = ckpt.load('public', 'x.npz')
      np.testing.assert_array_equal(loaded['a'], obj['a'])
      np.testing.assert_array_equal(loaded['b'], obj['b'])

  def test_cleanup_removes_files_and_tolerates_missing(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt = checkpoint_lib.Checkpointer(working_dir=tmpdir)
      ckpt.setup('sensitive')
      ckpt.save('sensitive', 'marginals.npz', {'a': jnp.arange(2)})
      self.assertIsNotNone(ckpt.load('sensitive', 'marginals.npz'))
      # Present and missing files can be requested together.
      ckpt.cleanup('sensitive', ['marginals.npz', 'missing.npz'])
      self.assertIsNone(ckpt.load('sensitive', 'marginals.npz'))

  def test_routes_io_through_filesystem(self):
    """Checkpointer performs all I/O through the provided FileSystem."""
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
      ckpt = checkpoint_lib.Checkpointer(working_dir=tmpdir, fs=fs)
      ckpt.setup('public')
      ckpt.save('public', 'x.npz', {'a': jnp.arange(1)})
      self.assertIsNotNone(ckpt.load('public', 'x.npz'))
      self.assertIn(('open', 'wb'), events)
      self.assertIn(('open', 'rb'), events)


if __name__ == '__main__':
  absltest.main()

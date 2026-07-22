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
from dpsynth import filesystem


class FileSystemTest(absltest.TestCase):

  def test_default_local_roundtrip(self):
    """Default FileSystem reads, writes, and removes on local disk."""
    fs = filesystem.FileSystem()
    with tempfile.TemporaryDirectory() as tmpdir:
      subdir = os.path.join(tmpdir, 'a', 'b')
      fs.makedirs(subdir)
      self.assertTrue(fs.exists(subdir))

      path = os.path.join(subdir, 'test.bin')
      self.assertFalse(fs.exists(path))

      with fs.open(path, 'wb') as f:
        f.write(b'hello')
      self.assertTrue(fs.exists(path))

      with fs.open(path, 'rb') as f:
        self.assertEqual(f.read(), b'hello')

      fs.remove(path)
      self.assertFalse(fs.exists(path))

  def test_custom_callables(self):
    """FileSystem dispatches to the provided callables."""
    calls = []
    fs = filesystem.FileSystem(
        open=lambda path, mode: calls.append(('open', path, mode)),
        exists=lambda path: bool(calls.append(('exists', path))),
        makedirs=lambda path: calls.append(('makedirs', path)),
        remove=lambda path: calls.append(('remove', path)),
    )
    fs.makedirs('/fake/dir')
    fs.exists('/fake/path')
    fs.open('/fake/file', 'rb')
    fs.remove('/fake/file')

    self.assertEqual(
        [c[0] for c in calls], ['makedirs', 'exists', 'open', 'remove']
    )


if __name__ == '__main__':
  absltest.main()

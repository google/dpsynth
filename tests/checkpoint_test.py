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

"""Unit tests for dpsynth.checkpoint."""

import dataclasses
import pathlib
from absl.testing import absltest
from dpsynth import api
from dpsynth import checkpoint as checkpoint_lib
from etils import epath
import jax.numpy as jnp
import mbi
import numpy as np


class CheckpointerTest(absltest.TestCase):

  def test_noop_when_disabled(self):
    ckpt = checkpoint_lib.Checkpointer(working_dir=None)
    self.assertIsNone(ckpt.path)
    self.assertFalse(ckpt.exists('model.npz'))
    self.assertIsNone(ckpt.load('model.npz'))

    # Saving should be a no-op and not raise.
    domain = mbi.Domain.fromdict({'a': 2, 'b': 3})
    cliques = [('a',), ('b',)]
    potentials = mbi.CliqueVector.zeros(domain, cliques)
    ckpt.save('model.npz', potentials)
    self.assertFalse(ckpt.exists('model.npz'))
    self.assertIsNone(ckpt.load('model.npz'))

  def test_save_and_load_roundtrip(self):
    working_dir = self.create_tempdir().full_path
    ckpt = checkpoint_lib.Checkpointer(working_dir=working_dir)

    domain = mbi.Domain.fromdict({'a': 2, 'b': 3})
    cliques = [('a',), ('a', 'b')]
    potentials = mbi.CliqueVector.zeros(domain, cliques)
    potentials[('a',)] = jnp.array([1.0, 2.0])
    marginals = mbi.CliqueVector.zeros(domain, cliques)
    mrf = mbi.MarkovRandomField(
        potentials=potentials, marginals=marginals, total=10.0
    )

    self.assertFalse(ckpt.exists('model.npz'))
    ckpt.save('model.npz', mrf)
    self.assertTrue(ckpt.exists('model.npz'))

    loaded = ckpt.load('model.npz')
    self.assertIsInstance(loaded, mbi.MarkovRandomField)
    np.testing.assert_allclose(loaded.potentials[('a',)], potentials[('a',)])
    self.assertEqual(loaded.total, 10.0)

  def test_save_and_load_linear_measurements(self):
    working_dir = self.create_tempdir().full_path
    ckpt = checkpoint_lib.Checkpointer(working_dir=working_dir)

    measurements = [
        mbi.LinearMeasurement(np.array([5.0, 10.0]), ('a',), stddev=1.0),
        mbi.LinearMeasurement(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), ('a', 'b'), stddev=0.5
        ),
    ]
    ckpt.save('measurements.npz', measurements)
    self.assertTrue(ckpt.exists('measurements.npz'))

    loaded = ckpt.load('measurements.npz')
    self.assertLen(loaded, 2)
    self.assertEqual(loaded[0].clique, ('a',))
    self.assertEqual(loaded[0].stddev, 1.0)
    np.testing.assert_allclose(
        loaded[0].noisy_measurement, measurements[0].noisy_measurement
    )
    self.assertEqual(loaded[1].clique, ('a', 'b'))
    self.assertEqual(loaded[1].stddev, 0.5)
    np.testing.assert_allclose(
        loaded[1].noisy_measurement, measurements[1].noisy_measurement
    )

  def test_load_nonexistent_returns_none(self):
    working_dir = self.create_tempdir().full_path
    ckpt = checkpoint_lib.Checkpointer(working_dir=working_dir)
    self.assertIsNone(ckpt.load('nonexistent.npz'))

  def test_accepts_different_path_types(self):
    temp_dir = self.create_tempdir().full_path

    # str
    ckpt_str = checkpoint_lib.Checkpointer(working_dir=temp_dir)
    ckpt_str.save('test_str.npz', {'v': jnp.array([1, 2, 3])})
    self.assertTrue(ckpt_str.exists('test_str.npz'))

    # pathlib.Path
    ckpt_pathlib = checkpoint_lib.Checkpointer(
        working_dir=pathlib.Path(temp_dir)
    )
    self.assertTrue(ckpt_pathlib.exists('test_str.npz'))
    loaded = ckpt_pathlib.load('test_str.npz')
    np.testing.assert_array_equal(loaded['v'], [1, 2, 3])

    # epath.Path
    ckpt_epath = checkpoint_lib.Checkpointer(working_dir=epath.Path(temp_dir))
    self.assertTrue(ckpt_epath.exists('test_str.npz'))

  def test_with_working_dir_propagates_if_supported(self):
    @dataclasses.dataclass(frozen=True)
    class MockConfig(api.MechanismConfig):
      working_dir: epath.PathLike | None = None

      def configure(self, *args, **kwargs):
        pass

    cfg = MockConfig()
    self.assertIsNone(cfg.working_dir)

    updated = cfg.with_working_dir('/tmp/test_dir')
    self.assertEqual(updated.working_dir, '/tmp/test_dir')

    # Existing working_dir is preserved (not overwritten).
    preserved = updated.with_working_dir('/tmp/other_dir')
    self.assertEqual(preserved.working_dir, '/tmp/test_dir')

  def test_with_working_dir_noop_on_unsupported_config(self):
    @dataclasses.dataclass(frozen=True)
    class MockConfigWithoutWorkingDir(api.MechanismConfig):
      param: int = 42

      def configure(self, *args, **kwargs):
        pass

    cfg = MockConfigWithoutWorkingDir()
    updated = cfg.with_working_dir('/tmp/test_dir')
    self.assertIs(updated, cfg)

  def test_with_working_dir_none_returns_self(self):
    @dataclasses.dataclass(frozen=True)
    class MockConfig(api.MechanismConfig):
      working_dir: epath.PathLike | None = None

      def configure(self, *args, **kwargs):
        pass

    cfg = MockConfig()
    self.assertIs(cfg.with_working_dir(None), cfg)


if __name__ == '__main__':
  absltest.main()

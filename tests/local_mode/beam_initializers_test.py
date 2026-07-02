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


from __future__ import annotations

from absl.testing import absltest
import apache_beam as beam
from dpsynth import domain
from dpsynth.local_mode import beam_initializers
from dpsynth.local_mode import initialization
import numpy as np

_TEST_RESULTS = []


def _store(x):
  _TEST_RESULTS.append(x)


def _init_spec_from_initializer(init):
  """Helper: builds an InitSpec from a calibrated initializer."""
  if isinstance(init, initialization.NumericalInitializer):
    return beam_initializers.InitSpec(
        beam_initializers.ColumnType.NUMERICAL,
        init.mechanism,
        init.attribute,
        grid_size=init.grid_size,
    )
  elif isinstance(init, initialization.CategoricalInitializer):
    return beam_initializers.InitSpec(
        beam_initializers.ColumnType.CATEGORICAL,
        init.mechanism,
        init.attribute,
    )
  elif isinstance(init, initialization.OpenSetCategoricalInitializer):
    return beam_initializers.InitSpec(
        beam_initializers.ColumnType.OPENSET,
        init.mechanism,
        init.attribute,
        min_count=init.min_count,
    )
  raise TypeError(type(init))


class NumericalHistogramTest(absltest.TestCase):

  def _run(self, rows, attr, grid_size=101):
    init = initialization.NumericalInitializer(
        name='x', num_partitions=4, attribute=attr, grid_size=grid_size
    ).calibrate(zcdp_rho=np.inf)
    spec = _init_spec_from_initializer(init)
    _TEST_RESULTS.clear()
    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_initializers.ComputeSufficientStats({'x': spec})
      )
      _ = stats | beam.combiners.ToDict() | beam.Map(_store)
    return dict(_TEST_RESULTS[0]['x'])

  def test_basic_histogram(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    rows = [{'x': 10}, {'x': 10}, {'x': 50}, {'x': 90}]
    counts = self._run(rows, attr)
    self.assertEqual(counts.get(10, 0), 2)
    self.assertEqual(counts.get(50, 0), 1)
    self.assertEqual(counts.get(90, 0), 1)
    self.assertEqual(sum(counts.values()), 4)

  def test_nan_clip_to_range_true(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=True
    )
    rows = [{'x': float('nan')}, {'x': None}, {'x': 50}]
    counts = self._run(rows, attr)
    self.assertEqual(counts.get(0, 0), 2)
    self.assertEqual(counts.get(50, 0), 1)
    self.assertEqual(sum(counts.values()), 3)

  def test_nan_clip_to_range_false(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=False
    )
    rows = [{'x': float('nan')}, {'x': 50}, {'x': 75}]
    counts = self._run(rows, attr)
    self.assertNotIn(0, counts)
    self.assertEqual(counts.get(50, 0), 1)
    self.assertEqual(counts.get(75, 0), 1)
    self.assertEqual(sum(counts.values()), 2)


class CategoricalCountsTest(absltest.TestCase):

  def test_basic_counts(self):
    attr = domain.CategoricalAttribute(
        possible_values=['unk', 'a', 'b', 'c'], out_of_domain_index=0
    )
    init = initialization.CategoricalInitializer(
        name='col', attribute=attr
    ).calibrate(zcdp_rho=np.inf)
    spec = _init_spec_from_initializer(init)
    rows = [
        {'col': 'a'},
        {'col': 'a'},
        {'col': 'b'},
        {'col': 'c'},
        {'col': 'c'},
        {'col': 'c'},
        {'col': 'z'},  # unknown → mapped to 'unk' (index 0)
    ]
    _TEST_RESULTS.clear()
    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_initializers.ComputeSufficientStats({'col': spec})
      )
      _ = stats | beam.combiners.ToDict() | beam.Map(_store)
    counts = dict(_TEST_RESULTS[0]['col'])
    self.assertEqual(counts.get(0, 0), 1)
    self.assertEqual(counts.get(1, 0), 2)
    self.assertEqual(counts.get(2, 0), 1)
    self.assertEqual(counts.get(3, 0), 3)
    self.assertEqual(sum(counts.values()), 7)


class OpenSetCountsTest(absltest.TestCase):

  def test_basic_counts(self):
    attr = domain.OpenSetCategoricalAttribute(default_value=None)
    init = initialization.OpenSetCategoricalInitializer(
        name='col', attribute=attr, delta=0.01, min_count=1
    ).calibrate(zcdp_rho=np.inf)
    spec = _init_spec_from_initializer(init)
    rows = [
        {'col': 'apple'},
        {'col': 'apple'},
        {'col': 'banana'},
        {'col': 'cherry'},
        {'col': 'cherry'},
        {'col': 'cherry'},
    ]
    _TEST_RESULTS.clear()
    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_initializers.ComputeSufficientStats({'col': spec})
      )
      _ = stats | beam.combiners.ToDict() | beam.Map(_store)
    counts = dict(_TEST_RESULTS[0]['col'])
    self.assertEqual(counts['apple'], 2)
    self.assertEqual(counts['banana'], 1)
    self.assertEqual(counts['cherry'], 3)
    self.assertEqual(sum(counts.values()), 6)


class BeamInitializeTest(absltest.TestCase):

  def _make_init_specs(self):
    inits = {
        'score': (
            initialization.NumericalInitializer(
                name='score',
                num_partitions=4,
                attribute=domain.NumericalAttribute(min_value=0, max_value=100),
            ).calibrate(zcdp_rho=np.inf)
        ),
        'grade': (
            initialization.CategoricalInitializer(
                name='grade',
                attribute=domain.CategoricalAttribute(
                    possible_values=['a', 'b']
                ),
            ).calibrate(zcdp_rho=np.inf)
        ),
        'tag': (
            initialization.OpenSetCategoricalInitializer(
                name='tag',
                attribute=domain.OpenSetCategoricalAttribute(
                    default_value=None
                ),
                delta=0.01,
                min_count=1,
            ).calibrate(zcdp_rho=np.inf)
        ),
    }
    return {k: _init_spec_from_initializer(v) for k, v in inits.items()}

  def _run_pipeline(self, init_specs):
    rows = [
        {'score': 25.0, 'grade': 'a', 'tag': 'p'},
        {'score': 50.0, 'grade': 'b', 'tag': 'q'},
        {'score': 75.0, 'grade': 'a', 'tag': 'p'},
    ]
    rng = np.random.default_rng(42)
    _TEST_RESULTS.clear()
    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create(rows)
          | beam_initializers.BeamInitialize(init_specs, rng)
      )
      _ = result | beam.Map(_store)
    return _TEST_RESULTS[0]

  def test_end_to_end_mixed(self):
    results = self._run_pipeline(self._make_init_specs())
    self.assertLen(results, 3)
    for br in results.values():
      self.assertIsInstance(br, beam_initializers.BeamColumnResult)
    self.assertEqual(
        results['score'].column_type,
        beam_initializers.ColumnType.NUMERICAL,
    )
    self.assertEqual(
        results['grade'].column_type,
        beam_initializers.ColumnType.CATEGORICAL,
    )
    self.assertEqual(
        results['tag'].column_type,
        beam_initializers.ColumnType.OPENSET,
    )


if __name__ == '__main__':
  absltest.main()

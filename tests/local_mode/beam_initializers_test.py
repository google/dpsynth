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
import mbi
import numpy as np

_TEST_RESULTS = []


def _store(x):
  _TEST_RESULTS.append(x)


class NumericalHistogramTest(absltest.TestCase):

  def _run(self, rows, attr, grid_size=101):
    init = initialization.NumericalInitializer(
        name='x',
        num_partitions=4,
        attribute=attr,
        grid_size=grid_size,
    ).calibrate(zcdp_rho=np.inf)
    _TEST_RESULTS.clear()
    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_initializers.ComputeSufficientStats({'x': init})
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
        possible_values=['unk', 'a', 'b', 'c'],
        out_of_domain_index=0,
    )
    init = initialization.CategoricalInitializer(
        name='col',
        attribute=attr,
    ).calibrate(zcdp_rho=np.inf)
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
          | beam_initializers.ComputeSufficientStats({'col': init})
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
          | beam_initializers.ComputeSufficientStats({'col': init})
      )
      _ = stats | beam.combiners.ToDict() | beam.Map(_store)
    counts = dict(_TEST_RESULTS[0]['col'])
    self.assertEqual(counts['apple'], 2)
    self.assertEqual(counts['banana'], 1)
    self.assertEqual(counts['cherry'], 3)
    self.assertEqual(sum(counts.values()), 6)


class BeamInitializeTest(absltest.TestCase):

  def test_end_to_end_mixed(self):
    num_attr = domain.NumericalAttribute(min_value=0, max_value=100)
    cat_attr = domain.CategoricalAttribute(possible_values=['a', 'b'])
    open_attr = domain.OpenSetCategoricalAttribute(default_value=None)

    initializers = {
        'score': (
            initialization.NumericalInitializer(
                name='score', num_partitions=4, attribute=num_attr
            ).calibrate(zcdp_rho=np.inf)
        ),
        'grade': (
            initialization.CategoricalInitializer(
                name='grade', attribute=cat_attr
            ).calibrate(zcdp_rho=np.inf)
        ),
        'tag': (
            initialization.OpenSetCategoricalInitializer(
                name='tag', attribute=open_attr, delta=0.01, min_count=1
            ).calibrate(zcdp_rho=np.inf)
        ),
    }

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
          | beam_initializers.BeamInitialize(initializers, rng)
      )
      _ = result | beam.Map(_store)
    measurements = _TEST_RESULTS[0]

    self.assertLen(measurements, 3)
    for cm in measurements.values():
      self.assertIsInstance(cm, initialization.ColumnMeasurement)


class ComputeMarginalsTest(absltest.TestCase):

  def test_marginals_match_manual_counts(self):
    cat_attr = domain.CategoricalAttribute(possible_values=['a', 'b', 'c'])
    num_attr = domain.NumericalAttribute(min_value=0, max_value=10)
    cat_init = initialization.CategoricalInitializer(
        name='color',
        attribute=cat_attr,
    ).calibrate(zcdp_rho=np.inf)
    num_init = initialization.NumericalInitializer(
        name='size',
        num_partitions=4,
        attribute=num_attr,
        grid_size=11,
    ).calibrate(zcdp_rho=np.inf)
    domains = {'color': cat_attr, 'size': num_attr}
    rows = [
        {'color': 'a', 'size': 0},
        {'color': 'a', 'size': 5},
        {'color': 'b', 'size': 5},
        {'color': 'b', 'size': 10},
        {'color': 'c', 'size': 0},
        {'color': 'c', 'size': 0},
    ]

    # Stage 1: get ColumnMeasurements.
    inits = {'color': cat_init, 'size': num_init}
    rng = np.random.default_rng(42)
    _TEST_RESULTS.clear()
    with beam.Pipeline() as p:
      stats = (
          p
          | 'Create1' >> beam.Create(rows)
          | beam_initializers.ComputeSufficientStats(inits)
      )
      _ = stats | 'ToDict1' >> beam.combiners.ToDict() | beam.Map(_store)
    cms = beam_initializers.run_from_summary(_TEST_RESULTS[0], inits, rng)

    # Stage 2: compute marginals.
    workload = [('color',), ('size',), ('color', 'size')]
    _TEST_RESULTS.clear()
    with beam.Pipeline() as p:
      result = (
          p
          | 'Create2' >> beam.Create(rows)
          | beam_initializers.ComputeMarginals(cms, domains, workload)
      )
      _ = result | beam.Map(_store)

    cv = _TEST_RESULTS[0]
    self.assertIsInstance(cv, mbi.CliqueVector)
    self.assertLen(cv.cliques, 3)

    # 1-way: color [a=2, b=2, c=2].
    np.testing.assert_array_equal(
        cv.arrays[('color',)].datavector(),
        [2, 2, 2],
    )
    # 1-way: size total equals number of rows.
    self.assertEqual(cv.arrays[('size',)].datavector().sum(), 6)
    # 2-way: shape matches product of column sizes, total equals rows.
    joint = cv.arrays[('color', 'size')]
    expected_size = cms['color'].categorical_attribute.size
    expected_size *= cms['size'].categorical_attribute.size
    self.assertEqual(joint.domain.size(), expected_size)
    self.assertEqual(joint.datavector().sum(), 6)


if __name__ == '__main__':
  absltest.main()

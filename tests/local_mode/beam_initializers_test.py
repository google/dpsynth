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

import collections

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
from dpsynth import domain
from dpsynth.local_mode import beam_initializers
from dpsynth.local_mode import initialization
import mbi
import numpy as np


class NumericalHistogramTest(absltest.TestCase):

  def _run_and_assert(
      self, rows, attr, assert_fn, max_grid_size=101, num_partitions=4
  ):
    init = initialization.NumericalInitializer(
        name='x',
        num_partitions=num_partitions,
        attribute=attr,
        max_grid_size=max_grid_size,
    ).configure(zcdp_rho=np.inf)
    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_initializers.ComputeSufficientStats({'x': init})
      )
      beam_testing_util.assert_that(stats | beam.combiners.ToDict(), assert_fn)

  def _ref_counts(self, values, attr, max_grid_size=101, num_partitions=4):
    """In-memory grid histogram as an {index: count} dict."""
    init = initialization.NumericalInitializer(
        name='x',
        num_partitions=num_partitions,
        attribute=attr,
        max_grid_size=max_grid_size,
    ).configure(zcdp_rho=np.inf)
    dense = init._grid_histogram(np.asarray(values, dtype=float))
    return {i: int(c) for i, c in enumerate(dense) if c}

  def test_basic_histogram(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    rows = [{'x': 10}, {'x': 10}, {'x': 50}, {'x': 90}]

    def check(actual):
      counts = dict(actual[0]['x'])

      self.assertEqual(counts.get(10, 0), 2)
      self.assertEqual(counts.get(50, 0), 1)
      self.assertEqual(counts.get(90, 0), 1)
      self.assertEqual(sum(counts.values()), 4)

    self._run_and_assert(rows, attr, check)

  def test_nan_clip_to_range_true(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=True
    )
    rows = [{'x': float('nan')}, {'x': None}, {'x': 50}]

    def check(actual):
      counts = dict(actual[0]['x'])

      self.assertEqual(counts.get(0, 0), 2)
      self.assertEqual(counts.get(50, 0), 1)
      self.assertEqual(sum(counts.values()), 3)

    self._run_and_assert(rows, attr, check)

  def test_nan_clip_to_range_false(self):
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=False
    )
    rows = [{'x': float('nan')}, {'x': 50}, {'x': 75}]

    def check(actual):
      counts = dict(actual[0]['x'])

      self.assertNotIn(0, counts)
      self.assertEqual(counts.get(50, 0), 1)
      self.assertEqual(counts.get(75, 0), 1)
      self.assertEqual(sum(counts.values()), 2)

    self._run_and_assert(rows, attr, check)

  def test_beam_matches_in_memory_clip_true_nan(self):
    # clip_to_range=True: NaN/None fold into the minimum bin; nothing dropped.
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=True
    )
    values = [float('nan'), None, 50]
    ref_counts = self._ref_counts(values, attr)

    def check(actual):
      beam_counts = dict(actual[0]['x'])

      self.assertEqual(beam_counts, ref_counts)
      self.assertEqual(ref_counts, {0: 2, 50: 1})

    self._run_and_assert([{'x': v} for v in values], attr, check)

  def test_beam_matches_in_memory_clip_false_drops_ood(self):
    # clip_to_range=False: NaN and values outside [min, max] are dropped.
    attr = domain.NumericalAttribute(
        min_value=0, max_value=100, clip_to_range=False
    )
    values = [float('nan'), -5, 150, 50, 75]
    ref_counts = self._ref_counts(values, attr)

    def check(actual):
      beam_counts = dict(actual[0]['x'])

      self.assertEqual(beam_counts, ref_counts)
      self.assertEqual(ref_counts, {50: 1, 75: 1})

    self._run_and_assert([{'x': v} for v in values], attr, check)

  def test_beam_matches_in_memory_int_at_max_value(self):
    # Regression: an integer value at max_value used to yield a grid index of
    # grid_size (IndexError) when the grid step > 1. Both paths must now fold
    # it into the top bin and agree.
    attr = domain.NumericalAttribute(min_value=0, max_value=100, dtype='int')
    values = [0, 50, 100, 100]
    ref_counts = self._ref_counts(
        values, attr, max_grid_size=2, num_partitions=1
    )

    def check(actual):
      beam_counts = dict(actual[0]['x'])

      self.assertEqual(beam_counts, ref_counts)
      self.assertEqual(sum(beam_counts.values()), 4)

    self._run_and_assert(
        [{'x': v} for v in values],
        attr,
        check,
        max_grid_size=2,
        num_partitions=1,
    )


class CategoricalCountsTest(absltest.TestCase):

  def test_basic_counts(self):
    attr = domain.CategoricalAttribute(
        possible_values=['unk', 'a', 'b', 'c'],
        out_of_domain_index=0,
    )
    init = initialization.CategoricalInitializer(
        name='col',
        attribute=attr,
    ).configure(zcdp_rho=np.inf)
    rows = [
        {'col': 'a'},
        {'col': 'a'},
        {'col': 'b'},
        {'col': 'c'},
        {'col': 'c'},
        {'col': 'c'},
        {'col': 'z'},  # unknown → mapped to 'unk' (index 0)
    ]

    def check(actual):
      counts = dict(actual[0]['col'])

      self.assertEqual(counts.get(0, 0), 1)
      self.assertEqual(counts.get(1, 0), 2)
      self.assertEqual(counts.get(2, 0), 1)
      self.assertEqual(counts.get(3, 0), 3)
      self.assertEqual(sum(counts.values()), 7)

    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_initializers.ComputeSufficientStats({'col': init})
      )
      beam_testing_util.assert_that(stats | beam.combiners.ToDict(), check)


class OpenSetCountsTest(absltest.TestCase):

  def test_basic_counts(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')
    init = initialization.OpenSetCategoricalInitializer(
        name='col', attribute=attr, delta=0.01, min_count=1
    ).configure(zcdp_rho=np.inf)
    rows = [
        {'col': 'apple'},
        {'col': 'apple'},
        {'col': 'banana'},
        {'col': 'cherry'},
        {'col': 'cherry'},
        {'col': 'cherry'},
    ]

    def check(actual):
      counts = dict(actual[0]['col'])

      self.assertEqual(counts['apple'], 2)
      self.assertEqual(counts['banana'], 1)
      self.assertEqual(counts['cherry'], 3)
      self.assertEqual(sum(counts.values()), 6)

    with beam.Pipeline() as p:
      stats = (
          p
          | beam.Create(rows)
          | beam_initializers.ComputeSufficientStats({'col': init})
      )
      beam_testing_util.assert_that(stats | beam.combiners.ToDict(), check)


class BeamInitializeTest(absltest.TestCase):

  def test_end_to_end_mixed(self):
    num_attr = domain.NumericalAttribute(min_value=0, max_value=100)
    cat_attr = domain.CategoricalAttribute(possible_values=['a', 'b'])
    open_attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')

    initializers = {
        'score': (
            initialization.NumericalInitializer(
                name='score', num_partitions=4, attribute=num_attr
            ).configure(zcdp_rho=np.inf)
        ),
        'grade': (
            initialization.CategoricalInitializer(
                name='grade', attribute=cat_attr
            ).configure(zcdp_rho=np.inf)
        ),
        'tag': (
            initialization.OpenSetCategoricalInitializer(
                name='tag', attribute=open_attr, delta=0.01, min_count=1
            ).configure(zcdp_rho=np.inf)
        ),
    }

    rows = [
        {'score': 25.0, 'grade': 'a', 'tag': 'p'},
        {'score': 50.0, 'grade': 'b', 'tag': 'q'},
        {'score': 75.0, 'grade': 'a', 'tag': 'p'},
    ]
    rng = np.random.default_rng(42)

    def check(actual):
      measurements = actual[0]

      self.assertLen(measurements, 3)
      for cm in measurements.values():
        self.assertIsInstance(cm, initialization.ColumnMeasurement)

    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create(rows)
          | beam_initializers.BeamInitialize(initializers, rng)
      )
      beam_testing_util.assert_that(result, check)


class ComputeMarginalsTest(absltest.TestCase):

  def test_marginals_match_manual_counts(self):
    cat_attr = domain.CategoricalAttribute(possible_values=['a', 'b', 'c'])
    num_attr = domain.NumericalAttribute(min_value=0, max_value=10)
    cat_init = initialization.CategoricalInitializer(
        name='color',
        attribute=cat_attr,
    ).configure(zcdp_rho=np.inf)
    num_init = initialization.NumericalInitializer(
        name='size',
        num_partitions=4,
        attribute=num_attr,
        max_grid_size=11,
    ).configure(zcdp_rho=np.inf)
    domains = {'color': cat_attr, 'size': num_attr}
    rows = [
        {'color': 'a', 'size': 0},
        {'color': 'a', 'size': 5},
        {'color': 'b', 'size': 5},
        {'color': 'b', 'size': 10},
        {'color': 'c', 'size': 0},
        {'color': 'c', 'size': 0},
    ]

    # Stage 1: get ColumnMeasurements natively in Python.
    inits = {'color': cat_init, 'size': num_init}
    rng = np.random.default_rng(42)

    encoder = beam_initializers._EncodeColumns(inits)
    counts = collections.defaultdict(int)
    for row in rows:
      for encoded in encoder.process(row):
        counts[encoded] += 1

    summary = collections.defaultdict(list)
    for (col, val), count in counts.items():
      summary[col].append((val, count))

    compute_stats = beam_initializers.ComputeSufficientStats(inits)
    summary = {
        col: beam_initializers._filter_openset(
            (col, pairs), min_counts=compute_stats._openset_min_counts
        )[1]
        for col, pairs in summary.items()
    }
    cms = beam_initializers.run_from_summary(summary, inits, rng)

    # Stage 2: compute marginals.
    workload = [('color',), ('size',), ('color', 'size')]

    def check(actual):
      cv = actual[0]

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

    with beam.Pipeline() as p:
      result = (
          p
          | 'Create2' >> beam.Create(rows)
          | beam_initializers.ComputeMarginals(cms, domains, workload)
      )
      beam_testing_util.assert_that(result, check)


if __name__ == '__main__':
  absltest.main()

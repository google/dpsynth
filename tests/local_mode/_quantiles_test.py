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

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth.local_mode import _quantiles
import numpy as np


class QuantilesFromHistogramTest(parameterized.TestCase):

  def test_no_levels_returns_empty(self):
    rng = np.random.default_rng(0)
    counts = np.array([10])
    for jitter_strategy in ('symmetric', 'refine'):
      edges = _quantiles.quantiles_from_histogram(
          rng, counts, np.array([]), jitter_strategy
      )
      self.assertEmpty(edges)

  @parameterized.product(
      levels=(1, 2, 3, 4),
      jitter_strategy=('symmetric', 'refine'),
  )
  def test_edge_count_matches_levels(self, levels, jitter_strategy):
    rng = np.random.default_rng(0)
    grid_size = 10001
    counts = rng.integers(0, 20, size=grid_size)
    edges = _quantiles.quantiles_from_histogram(
        rng,
        counts,
        epsilon_levels=np.ones(levels),
        jitter_strategy=jitter_strategy,
    )
    self.assertLen(edges, 2**levels - 1)

  @parameterized.parameters(1, 2, 3, 4)
  def test_edge_count_matches_levels_with_spike(self, levels):
    # A large tied spike must not collapse split ranges and drop edges. With
    # whole-cell splits this dropped edges; jitter breaks up the spike so the
    # recursion always emits the full 2**levels - 1 edges.
    counts = np.zeros(101, dtype=np.int64)
    counts[:40] = 1
    counts[40] = 1000
    counts[41:80] = 1
    edges = _quantiles.quantiles_from_histogram(
        np.random.default_rng(0),
        counts,
        epsilon_levels=np.array([np.inf] * levels),
        jitter_strategy='refine',
    )
    self.assertLen(edges, 2**levels - 1)

  def test_integer_edges_are_integer_indices(self):
    # Integer attributes must return integer cell indices into ``counts``.
    counts = np.zeros(101, dtype=np.int64)
    counts[40] = 5000
    counts[:40] = 50
    counts[41:] = 50
    edges = _quantiles.quantiles_from_histogram(
        np.random.default_rng(0),
        counts,
        epsilon_levels=np.array([np.inf] * 3),
        jitter_strategy='refine',
    )
    for edge in edges:
      self.assertEqual(edge, int(edge))
      self.assertBetween(edge, 0, counts.size - 1)

  def test_exact_budget_matches_numpy_smooth(self):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 100, size=50000)
    counts = np.bincount(data, minlength=101)
    edges = _quantiles.quantiles_from_histogram(
        rng,
        counts,
        epsilon_levels=np.array([np.inf] * 3),
        jitter_strategy='refine',
    )
    # Cell indices map 1:1 to values here (delta == 1), so compare directly.
    expected = np.quantile(data, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    np.testing.assert_allclose(edges, expected, atol=1.0)

  def test_exact_budget_matches_numpy_with_spike(self):
    # Reproduces the failure mode from the 'hours-per-week' column of the adult
    # census dataset: a dominant spike at 40 carrying ~45% of the mass, with
    # lighter integer support on either side. The pre-jitter whole-cell split
    # lumped all tied mass into one subtree, biasing the low quantiles (e.g. the
    # 0.25 edge came out as 18 instead of 40) and dropping upper edges. Jitter
    # breaks up the spike so the recursive medians match numpy.
    below = np.arange(1, 40).repeat(230)
    spike = np.full(13500, 40)
    above = np.arange(41, 80).repeat(190)
    data = np.concatenate([below, spike, above])
    counts = np.bincount(data, minlength=101)
    edges = _quantiles.quantiles_from_histogram(
        np.random.default_rng(0),
        counts,
        epsilon_levels=np.array([np.inf] * 3),
        jitter_strategy='refine',
    )
    # Cell indices map 1:1 to values here (delta == 1), so compare directly.
    expected = np.quantile(data, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    np.testing.assert_allclose(edges, expected, atol=1.0)

  def test_spike_owns_consecutive_edges(self):
    # When a single value holds a majority of the mass, the quantiles on both
    # sides of the median should collapse onto that value. This is the key
    # correctness property the deterministic whole-cell split got wrong.
    counts = np.zeros(101, dtype=np.int64)
    counts[:40] = 20
    counts[40] = 20000  # ~96% of the mass.
    counts[41:80] = 20
    edges = _quantiles.quantiles_from_histogram(
        np.random.default_rng(0),
        counts,
        epsilon_levels=np.array([np.inf] * 3),
        jitter_strategy='refine',
    )
    # The interior quantiles (0.25 through 0.75) must all be cell 40.
    self.assertEqual(edges[1:6], [40, 40, 40, 40, 40])


if __name__ == '__main__':
  absltest.main()

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
import dp_accounting
from dpsynth import domain
from dpsynth.local_mode import _quantiles
from dpsynth.local_mode import initialization
from dpsynth.local_mode import vectorized_transformations as vtx
import numpy as np


class InitializationTest(absltest.TestCase):

  def test_numerical_initializer_dp_event(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=4, attribute=attr
    )
    event = initializer.configure(zcdp_rho=1.0).dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    self.assertLen(event.events, 2)
    for e in event.events:
      self.assertIsInstance(e, dp_accounting.ExponentialMechanismDpEvent)

  def test_numerical_initializer_call(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=4, attribute=attr
    )

    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    measurement = initializer.configure(zcdp_rho=np.inf)(rng, data)

    self.assertIsInstance(measurement, initialization.ColumnMeasurement)
    self.assertEqual(measurement.categorical_attribute.size, 4)
    self.assertIsNone(measurement.measurement)
    self.assertIsNotNone(measurement.bin_edges)

    encoded_data = vtx.discretize(data, measurement.bin_edges, attr)
    counts = np.bincount(encoded_data)

    # All 9 data points assigned to exactly 4 bins, each non-empty.
    self.assertEqual(counts.sum(), 9)
    self.assertLen(counts, 4)
    self.assertTrue(np.all(counts > 0))

  def test_numerical_initializer_deduplicates_bin_edges(self):
    """Concentrated data can make quantiles return duplicate edges."""
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    rng = np.random.default_rng(42)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=8, attribute=attr
    )
    # Data is heavily concentrated at 50.
    data = np.array([50] * 100 + [1, 99])
    result = initializer.configure(zcdp_rho=1.0)(rng, data)

    # After dedup, bin_edges should be strictly increasing.
    edges = result.bin_edges
    self.assertTrue(
        np.all(np.diff(edges) > 0),
        f'bin_edges not strictly increasing: {edges}',
    )
    # And downstream discretize should not crash.
    encoded = vtx.discretize(data, edges, attr)
    self.assertEqual(encoded.shape, data.shape)

  def test_numerical_initializer_integer_data(self):
    """Integer data within a narrow range can collapse quantile edges."""
    attr = domain.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=8, attribute=attr
    )
    # Only 3 distinct values but 8 partitions requested.
    data = np.array([3, 3, 3, 3, 5, 5, 5, 7])
    result = initializer.configure(zcdp_rho=1.0)(rng, data)

    edges = result.bin_edges
    self.assertTrue(
        np.all(np.diff(edges) > 0),
        f'bin_edges not strictly increasing: {edges}',
    )
    # Domain size may be < 8 due to dedup, but must be >= 2.
    self.assertGreaterEqual(result.categorical_attribute.size, 2)

  def test_numerical_initializer_integer_edges_are_floored(self):
    """Integer attributes should produce integer-valued bin edges."""
    attr = domain.NumericalAttribute(min_value=0, max_value=100, dtype='int')
    rng = np.random.default_rng(42)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=4, attribute=attr
    )
    data = np.arange(100)
    result = initializer.configure(zcdp_rho=100.0)(rng, data)
    # All edges should be integers (integer grid produces exact integer edges).
    np.testing.assert_array_equal(result.bin_edges, np.floor(result.bin_edges))
    # Edges must be within [min_value, max_value - 1].
    self.assertGreaterEqual(result.bin_edges[0], 0)
    self.assertLess(result.bin_edges[-1], 100)

  def test_numerical_initializer_measurement_with_merged_bins(self):
    """When integer edges collapse, merged bins get proportionally more mass."""
    attr = domain.NumericalAttribute(min_value=0, max_value=100, dtype='int')
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=8, attribute=attr
    )
    # Concentrated data will cause edge collisions.
    data = np.array([50] * 100 + [1, 99])
    result = initializer.configure(zcdp_rho=1.0)(
        rng, data, estimated_total=100.0
    )
    self.assertIsNotNone(result.measurement)
    # Measurement probabilities should sum to 1.0.
    np.testing.assert_allclose(
        result.measurement.noisy_measurement.sum(), 1.0, atol=1e-10
    )

  def test_max_grid_size_below_two_raises(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    for bad in (0, 1):
      with self.assertRaises(ValueError):
        initialization.NumericalInitializer(
            name='x', num_partitions=1, attribute=attr, max_grid_size=bad
        )

  def test_max_grid_size_two_int(self):
    rng = np.random.default_rng(42)
    attr = domain.NumericalAttribute(min_value=0, max_value=100, dtype='int')
    init = initialization.NumericalInitializer(
        name='x', num_partitions=1, attribute=attr, max_grid_size=2
    ).configure(zcdp_rho=1.0)
    self.assertEqual(init.grid_size, 2)
    result = init(rng, np.arange(100))
    self.assertIsNotNone(result.categorical_attribute)

  def test_max_grid_size_two_float(self):
    rng = np.random.default_rng(42)
    attr = domain.NumericalAttribute(min_value=0.0, max_value=100.0)
    init = initialization.NumericalInitializer(
        name='x', num_partitions=1, attribute=attr, max_grid_size=2
    ).configure(zcdp_rho=1.0)
    self.assertEqual(init.grid_size, 2)
    result = init(rng, np.linspace(0, 100, 100))
    self.assertIsNotNone(result.categorical_attribute)

  def test_int_grid_reserves_budget_for_jitter_refinement(self):
    # A wide integer range with many partitions must not blow past
    # max_grid_size once the m-fold jitter refinement is applied.
    attr = domain.NumericalAttribute(
        min_value=0, max_value=10_000_000, dtype='int'
    )
    max_grid_size = 100_000
    init = initialization.NumericalInitializer(
        name='x', num_partitions=64, attribute=attr, max_grid_size=max_grid_size
    )
    m = _quantiles.jitter_factor(init.num_partitions)
    self.assertLessEqual(init.grid_size * m, max_grid_size)

  def test_numerical_initializer_measurement_with_estimated_total(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializer(
        name='num_col', num_partitions=4, attribute=attr
    )
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    result = initializer.configure(zcdp_rho=1.0)(
        rng, data, estimated_total=100.0
    )

    self.assertIsNotNone(result.measurement)
    # Measurement should be uniform probabilities: 1.0 / num_bins each.
    num_bins = result.categorical_attribute.size
    expected_prob = 1.0 / num_bins
    np.testing.assert_allclose(
        result.measurement.noisy_measurement,
        np.full(num_bins, expected_prob),
    )
    self.assertEqual(result.measurement.clique, ('num_col',))
    # stddev should be 1/(sqrt(rho) * estimated_total) = 1/(1.0 * 100) = 0.01
    self.assertAlmostEqual(result.measurement.stddev, 0.01)

  def test_numerical_initializer_no_measurement_without_estimated_total(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=4, attribute=attr
    )
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    result = initializer.configure(zcdp_rho=1.0)(rng, data)
    self.assertIsNone(result.measurement)

  def test_integer_edges_at_max_value_absorbed_into_last_bin(self):
    """Edges at max_value are removed; their count goes to the last bin."""
    attr = domain.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=8, attribute=attr
    )
    # A spread of lower values carrying most of the mass, plus a moderate spike
    # at max_value.  The lower values form genuine interior bins, while the top
    # quantile edges land at max_value and must be absorbed into the last bin.
    data = np.concatenate([np.repeat(np.arange(0, 10), 20), np.full(100, 10)])
    result = initializer.configure(zcdp_rho=100.0)(
        rng, data, estimated_total=len(data)
    )
    # No edge should equal max_value (they get absorbed).
    if len(result.bin_edges) > 0:
      self.assertLess(result.bin_edges[-1], 10)
    # Measurement probabilities must still sum to 1.0.
    np.testing.assert_allclose(
        result.measurement.noisy_measurement.sum(), 1.0, atol=1e-10
    )
    # The last bin absorbs the max_value spike, so it should dominate.
    counts = result.measurement.noisy_measurement
    self.assertGreater(counts.max(), 1.0 / len(counts))

  def test_bin_weights_sum_to_num_partitions(self):
    """bin_weights must always sum to num_partitions regardless of dedup."""
    attr = domain.NumericalAttribute(min_value=0, max_value=20, dtype='int')
    for seed in range(10):
      rng = np.random.default_rng(seed)
      initializer = initialization.NumericalInitializer(
          name='test', num_partitions=8, attribute=attr
      )
      data = np.array([5] * 50 + [15] * 50)
      result = initializer.configure(zcdp_rho=1.0)(
          rng, data, estimated_total=100.0
      )
      # Sum of measurement probabilities = 1.0.
      np.testing.assert_allclose(
          result.measurement.noisy_measurement.sum(),
          1.0,
          atol=1e-10,
          err_msg=f'seed={seed}: probabilities do not sum to 1',
      )

  def test_integer_jitter_prevents_spurious_splits(self):
    """Positive jitter should prevent edges from splitting across integers."""
    attr = domain.NumericalAttribute(min_value=0, max_value=100, dtype='int')
    rng = np.random.default_rng(42)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=4, attribute=attr
    )
    # Uniform data: with high budget, edges should land at 25, 50, 75.
    data = np.arange(101)
    result = initializer.configure(zcdp_rho=1000.0)(rng, data)
    # With 4 partitions and 3 edges, no dedup should be needed.
    self.assertLen(result.bin_edges, 3)
    # All edges should be integers.
    np.testing.assert_array_equal(result.bin_edges, np.floor(result.bin_edges))

  def test_integer_heterogeneous_data_buckets(self):
    """Heterogeneous integer data produces sensible bucket partitioning."""
    attr = domain.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    rng = np.random.default_rng(42)
    initializer = initialization.NumericalInitializer(
        name='x', num_partitions=4, attribute=attr
    )
    # Deliberately lumpy distribution: 45 points across 4 distinct values.
    data = np.array([0] * 10 + [3] * 10 + [5] * 17 + [6] * 8)
    result = initializer.configure(zcdp_rho=np.inf)(
        rng, data, estimated_total=len(data)
    )
    # Edges should be integers strictly inside [min_value, max_value).
    for e in result.bin_edges:
      self.assertEqual(e, int(e), f'edge {e} is not an integer')
      self.assertGreaterEqual(e, 0)
      self.assertLess(e, 10)

    # Discretize and verify every data point is assigned.
    encoded = vtx.discretize(data, result.bin_edges, attr)
    self.assertEqual(encoded.shape, data.shape)
    num_bins = result.categorical_attribute.size
    true_counts = np.bincount(encoded, minlength=num_bins)
    self.assertLen(data, true_counts.sum())
    # Bins covering the data range should be non-empty; trailing bins
    # beyond the data (e.g. [7, 10] when max data is 6) may be empty.
    self.assertGreater(
        np.count_nonzero(true_counts),
        1,
        f'too few occupied bins: counts={true_counts},'
        f' edges={result.bin_edges}',
    )


class MeasurementApproximationTest(parameterized.TestCase):
  """Property test: measurement counts ≈ true histogram.

  The heuristic uniform measurement should satisfy
      L1(measurement, true_histogram) < max(3 / sqrt(rho), 2 - 1/K)
  where the 3/sqrt(rho) term covers quantile noise and 2-1/K covers the
  worst-case uniform-vs-delta misspecification error (K = num_bins).
  """

  @parameterized.named_parameters(
      # --- High budget (rho=1000): misspecification-dominated ---
      dict(
          testcase_name='uniform_int',
          attr=domain.NumericalAttribute(
              min_value=0, max_value=50, dtype='int'
          ),
          data=np.arange(51),
          num_partitions=4,
          rho=1000.0,
      ),
      dict(
          testcase_name='heterogeneous_int',
          attr=domain.NumericalAttribute(
              min_value=0, max_value=10, dtype='int'
          ),
          data=np.array([0] * 10 + [3] * 10 + [5] * 17 + [6] * 8 + [9] * 5),
          num_partitions=4,
          rho=1000.0,
      ),
      dict(
          testcase_name='uniform_float',
          attr=domain.NumericalAttribute(
              min_value=0, max_value=100, dtype='float'
          ),
          data=np.linspace(0, 100, 200),
          num_partitions=8,
          rho=1000.0,
      ),
      dict(
          testcase_name='boundary_heavy_int',
          attr=domain.NumericalAttribute(
              min_value=0, max_value=100, dtype='int'
          ),
          data=np.array([0] * 30 + [100] * 30 + [50] * 40),
          num_partitions=4,
          rho=1000.0,
      ),
      dict(
          testcase_name='bimodal_int',
          attr=domain.NumericalAttribute(
              min_value=0, max_value=100, dtype='int'
          ),
          data=np.array([10] * 50 + [90] * 50),
          num_partitions=8,
          rho=1000.0,
      ),
      dict(
          testcase_name='sparse_int',
          attr=domain.NumericalAttribute(
              min_value=0, max_value=1000, dtype='int'
          ),
          data=np.array([1] * 40 + [500] * 10 + [999] * 50),
          num_partitions=4,
          rho=1000.0,
      ),
      # --- Low budget (rho=0.5): noise-dominated ---
      dict(
          testcase_name='uniform_int_low_rho',
          attr=domain.NumericalAttribute(
              min_value=0, max_value=50, dtype='int'
          ),
          data=np.arange(51),
          num_partitions=4,
          rho=0.5,
      ),
      dict(
          testcase_name='uniform_float_low_rho',
          attr=domain.NumericalAttribute(
              min_value=0, max_value=100, dtype='float'
          ),
          data=np.linspace(0, 100, 200),
          num_partitions=8,
          rho=0.5,
      ),
      dict(
          testcase_name='heterogeneous_int_low_rho',
          attr=domain.NumericalAttribute(
              min_value=0, max_value=10, dtype='int'
          ),
          data=np.array([0] * 10 + [3] * 10 + [5] * 17 + [6] * 8 + [9] * 5),
          num_partitions=4,
          rho=0.5,
      ),
  )
  def test_measurement_approximates_true_histogram(
      self, attr, data, num_partitions, rho
  ):
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializer(
        name='x', num_partitions=num_partitions, attribute=attr
    )
    result = initializer.configure(zcdp_rho=rho)(
        rng, data, estimated_total=len(data)
    )
    # -- Structural checks (must always hold) --
    num_bins = result.categorical_attribute.size
    self.assertGreaterEqual(num_bins, 2)
    measurement = result.measurement
    self.assertIsNotNone(measurement)
    # Measurement probabilities must sum to 1.0.
    np.testing.assert_allclose(
        measurement.noisy_measurement.sum(), 1.0, atol=1e-10
    )
    # All measurement probabilities should be positive.
    self.assertTrue(
        np.all(measurement.noisy_measurement > 0),
        'non-positive measurement probabilities:'
        f' {measurement.noisy_measurement}',
    )
    # -- Statistical approximation check --
    encoded = vtx.discretize(data, result.bin_edges, attr)
    true_counts = np.bincount(encoded, minlength=num_bins).astype(float)
    true_prob = true_counts / true_counts.sum()
    meas_prob = measurement.noisy_measurement
    l1_dist = np.abs(true_prob - meas_prob).sum()
    # 3/sqrt(rho) covers quantile noise; 2-1/K covers uniform-vs-delta.
    max_l1 = max(3.0 / np.sqrt(rho), 2.0 - 1.0 / num_bins)
    self.assertLess(
        l1_dist,
        max_l1,
        f'Measurement too far from true histogram (L1={l1_dist:.3f},'
        f' bound={max_l1:.3f}, rho={rho}):\n'
        f'  true_prob = {true_prob}\n'
        f'  meas_prob = {meas_prob}',
    )

  def test_measurement_property_random_configs(self):
    """Randomized property test: max(3/sqrt(rho), 2-1/K) over many configs."""
    master_rng = np.random.default_rng(20260619)
    num_trials = 50

    for trial in range(num_trials):
      with self.subTest(trial=trial):
        # -- Random configuration --
        rho = float(10 ** master_rng.uniform(-1, 3))  # 0.1 to 1000
        num_partitions = int(master_rng.choice([2, 4, 8, 16]))
        min_val = int(master_rng.integers(0, 50))
        max_val = min_val + int(master_rng.integers(5, 200))
        is_int = bool(master_rng.random() < 0.5)

        # -- Random data as a mixture of 1-5 point masses --
        num_modes = int(master_rng.integers(1, 6))
        modes = master_rng.integers(min_val, max_val + 1, size=num_modes)
        weights = master_rng.dirichlet(np.ones(num_modes))
        n = int(master_rng.integers(50, 300))
        counts = np.round(weights * n).astype(int)
        counts[-1] = n - counts[:-1].sum()  # ensure exact total
        data = np.concatenate([np.full(c, m) for c, m in zip(counts, modes)])
        if not is_int:
          data = data.astype(float)

        dtype = 'int' if is_int else 'float'
        attr = domain.NumericalAttribute(
            min_value=min_val, max_value=max_val, dtype=dtype
        )

        # -- Run initializer --
        rng = np.random.default_rng(trial)
        initializer = initialization.NumericalInitializer(
            name='x', num_partitions=num_partitions, attribute=attr
        )
        result = initializer.configure(zcdp_rho=rho)(
            rng, data, estimated_total=len(data)
        )

        # -- Structural checks --
        measurement = result.measurement
        self.assertIsNotNone(measurement)
        np.testing.assert_allclose(
            measurement.noisy_measurement.sum(),
            1.0,
            atol=1e-10,
            err_msg=f'trial={trial}: probabilities do not sum to 1',
        )
        self.assertTrue(
            np.all(measurement.noisy_measurement > 0),
            f'trial={trial}: non-positive probabilities '
            f'{measurement.noisy_measurement}',
        )

        # -- L1 property check --
        num_bins = result.categorical_attribute.size
        encoded = vtx.discretize(data, result.bin_edges, attr)
        true_counts = np.bincount(encoded, minlength=num_bins).astype(float)
        true_prob = true_counts / true_counts.sum()
        meas_prob = measurement.noisy_measurement
        l1_dist = np.abs(true_prob - meas_prob).sum()
        max_l1 = max(3.0 / np.sqrt(rho), 2.0 - 1.0 / num_bins)
        self.assertLess(
            l1_dist,
            max_l1,
            f'trial={trial} (rho={rho:.2f}, K={num_partitions},'
            f' modes={modes}, is_int={is_int}):\n'
            f'  L1={l1_dist:.3f}, bound={max_l1:.3f}\n'
            f'  true_prob={true_prob}\n'
            f'  meas_prob={meas_prob}',
        )


class CategoricalInitializerTest(absltest.TestCase):

  def test_dp_event(self):
    attr = domain.CategoricalAttribute(possible_values=['A', 'B', 'C'])
    initializer = initialization.CategoricalInitializer(
        name='test', attribute=attr
    )
    event = initializer.configure(zcdp_rho=0.5).dp_event
    self.assertIsInstance(event, dp_accounting.GaussianDpEvent)
    # rho = 0.5 => sigma = 1/sqrt(2*0.5) = 1.0
    self.assertEqual(event.noise_multiplier, 1.0)

  def test_call_noiseless(self):
    attr = domain.CategoricalAttribute(possible_values=['A', 'B', 'C'])
    rng = np.random.default_rng(0)
    initializer = initialization.CategoricalInitializer(
        name='col', attribute=attr
    )
    data = np.array(['A', 'A', 'B', 'C', 'C', 'C'])
    result = initializer.configure(zcdp_rho=np.inf)(rng, data)

    self.assertIsInstance(result, initialization.ColumnMeasurement)
    self.assertEqual(result.categorical_attribute, attr)
    self.assertIsNotNone(result.measurement)
    np.testing.assert_array_equal(
        result.measurement.noisy_measurement, [2, 1, 3]
    )
    self.assertEqual(result.measurement.clique, ('col',))
    self.assertEqual(result.measurement.stddev, 0.0)

  def test_out_of_domain_values(self):
    attr = domain.CategoricalAttribute(
        possible_values=['<OOD>', 'X', 'Y'], out_of_domain_index=0
    )
    rng = np.random.default_rng(0)
    initializer = initialization.CategoricalInitializer(
        name='col', attribute=attr
    )
    data = np.array(['X', 'Y', 'Z', 'W'])
    result = initializer.configure(zcdp_rho=np.inf)(rng, data)

    # 'Z' and 'W' are OOD, mapped to index 0 (None).
    np.testing.assert_array_equal(
        result.measurement.noisy_measurement, [2, 1, 1]
    )


class OpenSetCategoricalInitializerTest(absltest.TestCase):

  def test_dp_event(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')
    initializer = initialization.OpenSetCategoricalInitializer(
        name='test', attribute=attr, delta=1e-5
    )
    event = initializer.configure(zcdp_rho=0.5).dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    self.assertLen(event.events, 2)
    self.assertIsInstance(event.events[0], dp_accounting.GaussianDpEvent)
    self.assertIsInstance(
        event.events[1], dp_accounting.dp_event.EpsilonDeltaDpEvent
    )
    self.assertEqual(event.events[1].delta, 1e-5)

  def test_call_noiseless(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')
    rng = np.random.default_rng(42)
    initializer = initialization.OpenSetCategoricalInitializer(
        name='col', attribute=attr, delta=1e-5
    )
    # 'A' appears 100 times, 'B' 50, 'C' 1 (rare).
    data = np.array(['A'] * 100 + ['B'] * 50 + ['C'] * 1)
    result = initializer.configure(zcdp_rho=np.inf)(rng, data)

    self.assertIsInstance(result, initialization.ColumnMeasurement)
    self.assertIsNotNone(result.measurement)
    # With infinite budget, all values with count > 0 should be selected.
    discovered = set(result.categorical_attribute.possible_values)
    self.assertIn('A', discovered)
    self.assertIn('B', discovered)
    self.assertIn('<OOD>', discovered)  # default value always present
    # Default value is always first.
    self.assertEqual(result.categorical_attribute.possible_values[0], '<OOD>')
    self.assertEqual(result.categorical_attribute.out_of_domain_index, 0)

  def test_undiscovered_values_map_to_default(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='OTHER')
    rng = np.random.default_rng(0)
    initializer = initialization.OpenSetCategoricalInitializer(
        name='col', attribute=attr, delta=1e-5
    )
    data = np.array(['A'] * 100 + ['B'] * 50)
    result = initializer.configure(zcdp_rho=np.inf)(rng, data)

    cat_attr = result.categorical_attribute
    # Discovered values map to valid indices.
    encoded = vtx.discrete_encode(np.array(['A']), cat_attr)
    self.assertGreater(encoded[0], 0)
    # Unknown value maps to the out-of-domain (default) index at 0.
    self.assertEqual(cat_attr.out_of_domain_index, 0)
    encoded_z = vtx.discrete_encode(np.array(['Z']), cat_attr)
    self.assertEqual(encoded_z[0], 0)

  def test_empty_data(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='<OOD>')
    rng = np.random.default_rng(0)
    initializer = initialization.OpenSetCategoricalInitializer(
        name='col', attribute=attr, delta=1e-5
    )
    data = np.array([], dtype=str)
    result = initializer.configure(zcdp_rho=np.inf)(rng, data)

    # Only the default value should be in the domain.
    self.assertEqual(result.categorical_attribute.possible_values, ['<OOD>'])
    self.assertEqual(result.categorical_attribute.size, 1)


class NumericalInitializerFromSummaryTest(absltest.TestCase):
  """Tests that NumericalInitializer.from_summary produces valid results."""

  def test_calibrate_sets_dp_event(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    init = initialization.NumericalInitializer(
        name='age',
        num_partitions=4,
        max_grid_size=10001,
        attribute=attr,
    ).configure(zcdp_rho=1.0)
    event = init.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    # 4 partitions = 2 levels.
    self.assertLen(event.events, 2)

  def test_uncalibrated_raises(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    init = initialization.NumericalInitializer(
        name='age',
        num_partitions=4,
        attribute=attr,
    )
    with self.assertRaises(ValueError):
      init.from_summary(np.random.default_rng(0), np.zeros(100))

  def test_integer_attribute_snaps_edges(self):
    rng = np.random.default_rng(42)
    attr = domain.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    init = initialization.NumericalInitializer(
        name='count',
        num_partitions=4,
        attribute=attr,
    ).configure(zcdp_rho=1.0)
    # Integer grid: grid_size = 11 (one bin per integer 0..10).
    counts = rng.integers(0, 30, size=init.grid_size)
    cm = init.from_summary(rng, counts)
    for edge in cm.bin_edges:
      self.assertEqual(edge, int(edge))

  def test_call_and_from_summary_produce_same_structure(self):
    """Both entry points should produce valid ColumnMeasurements."""
    rng = np.random.default_rng(42)
    attr = domain.NumericalAttribute(min_value=0.0, max_value=100.0)
    max_grid_size = 10001
    init = initialization.NumericalInitializer(
        name='x',
        num_partitions=4,
        attribute=attr,
        max_grid_size=max_grid_size,
    ).configure(zcdp_rho=1.0)

    data = np.linspace(0.0, 100.0, 500)
    cm_call = init(rng, data, estimated_total=500.0)
    self.assertIsNotNone(cm_call.bin_edges)
    self.assertIsNotNone(cm_call.measurement)

    rng2 = np.random.default_rng(42)
    counts = np.random.randint(0, 100, size=max_grid_size)
    cm_summary = init.from_summary(rng2, counts, estimated_total=500.0)
    self.assertIsNotNone(cm_summary.bin_edges)
    self.assertIsNotNone(cm_summary.measurement)


if __name__ == '__main__':
  absltest.main()

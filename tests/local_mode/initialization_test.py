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
import dp_accounting
from dpsynth import domain
from dpsynth.local_mode import initialization
from dpsynth.local_mode import vectorized_transformations as vtx
import numpy as np


class InitializationTest(absltest.TestCase):

  def test_numerical_initializer_dp_event(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=4, attribute=attr
    )
    event = initializer.calibrate(zcdp_rho=1.0).dp_event
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
    # Level 0 median [1..9] --> 5
    # Level 1 medians: [1..5] --> 3, [6..9] --> 7.5
    measurement = initializer.calibrate(zcdp_rho=np.inf)(rng, data)

    self.assertIsInstance(measurement, initialization.ColumnMeasurement)
    self.assertEqual(measurement.categorical_attribute.size, 4)
    self.assertIsNone(measurement.measurement)
    self.assertIsNotNone(measurement.bin_edges)

    encoded_data = vtx.discretize(data, measurement.bin_edges, attr)
    counts = np.bincount(encoded_data)

    # Expected Partitioning: 1 2 3 | 4 5 | 6 7 | 8 9
    np.testing.assert_array_equal(counts, [3, 2, 2, 2])

  def test_numerical_initializer_deduplicates_bin_edges(self):
    """Concentrated data can make quantiles return duplicate edges."""
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    rng = np.random.default_rng(42)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=8, attribute=attr
    )
    # Data is heavily concentrated at 50.
    data = np.array([50] * 100 + [1, 99])
    result = initializer.calibrate(zcdp_rho=1.0)(rng, data)

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
    result = initializer.calibrate(zcdp_rho=1.0)(rng, data)

    edges = result.bin_edges
    self.assertTrue(
        np.all(np.diff(edges) > 0),
        f'bin_edges not strictly increasing: {edges}',
    )
    # Domain size may be < 8 due to dedup, but must be >= 2.
    self.assertGreaterEqual(result.categorical_attribute.size, 2)


class CategoricalInitializerTest(absltest.TestCase):

  def test_dp_event(self):
    attr = domain.CategoricalAttribute(possible_values=['A', 'B', 'C'])
    initializer = initialization.CategoricalInitializer(
        name='test', attribute=attr
    )
    event = initializer.calibrate(zcdp_rho=0.5).dp_event
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
    result = initializer.calibrate(zcdp_rho=np.inf)(rng, data)

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
        possible_values=[None, 'X', 'Y'], out_of_domain_index=0
    )
    rng = np.random.default_rng(0)
    initializer = initialization.CategoricalInitializer(
        name='col', attribute=attr
    )
    data = np.array(['X', 'Y', 'Z', 'W'])
    result = initializer.calibrate(zcdp_rho=np.inf)(rng, data)

    # 'Z' and 'W' are OOD, mapped to index 0 (None).
    np.testing.assert_array_equal(
        result.measurement.noisy_measurement, [2, 1, 1]
    )


class OpenSetCategoricalInitializerTest(absltest.TestCase):

  def test_dp_event(self):
    attr = domain.OpenSetCategoricalAttribute(default_value=None)
    initializer = initialization.OpenSetCategoricalInitializer(
        name='test', attribute=attr, delta=1e-5
    )
    event = initializer.calibrate(zcdp_rho=0.5).dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    self.assertLen(event.events, 2)
    self.assertIsInstance(event.events[0], dp_accounting.GaussianDpEvent)
    self.assertIsInstance(
        event.events[1], dp_accounting.dp_event.EpsilonDeltaDpEvent
    )
    self.assertEqual(event.events[1].delta, 1e-5)

  def test_call_noiseless(self):
    attr = domain.OpenSetCategoricalAttribute(default_value=None)
    rng = np.random.default_rng(42)
    initializer = initialization.OpenSetCategoricalInitializer(
        name='col', attribute=attr, delta=1e-5
    )
    # 'A' appears 100 times, 'B' 50, 'C' 1 (rare).
    data = np.array(['A'] * 100 + ['B'] * 50 + ['C'] * 1)
    result = initializer.calibrate(zcdp_rho=np.inf)(rng, data)

    self.assertIsInstance(result, initialization.ColumnMeasurement)
    self.assertIsNotNone(result.measurement)
    # With infinite budget, all values with count > 0 should be selected.
    discovered = set(result.categorical_attribute.possible_values)
    self.assertIn('A', discovered)
    self.assertIn('B', discovered)
    self.assertIn(None, discovered)  # default value always present
    # Default value is always first.
    self.assertIsNone(result.categorical_attribute.possible_values[0])
    self.assertEqual(result.categorical_attribute.out_of_domain_index, 0)

  def test_undiscovered_values_map_to_default(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='OTHER')
    rng = np.random.default_rng(0)
    initializer = initialization.OpenSetCategoricalInitializer(
        name='col', attribute=attr, delta=1e-5
    )
    data = np.array(['A'] * 100 + ['B'] * 50)
    result = initializer.calibrate(zcdp_rho=np.inf)(rng, data)

    cat_attr = result.categorical_attribute
    # Discovered values map to valid indices.
    encoded = vtx.discrete_encode(np.array(['A']), cat_attr)
    self.assertGreater(encoded[0], 0)
    # Unknown value maps to the out-of-domain (default) index at 0.
    self.assertEqual(cat_attr.out_of_domain_index, 0)
    encoded_z = vtx.discrete_encode(np.array(['Z']), cat_attr)
    self.assertEqual(encoded_z[0], 0)

  def test_empty_data(self):
    attr = domain.OpenSetCategoricalAttribute(default_value=None)
    rng = np.random.default_rng(0)
    initializer = initialization.OpenSetCategoricalInitializer(
        name='col', attribute=attr, delta=1e-5
    )
    data = np.array([], dtype=str)
    result = initializer.calibrate(zcdp_rho=np.inf)(rng, data)

    # Only the default value should be in the domain.
    self.assertEqual(result.categorical_attribute.possible_values, [None])
    self.assertEqual(result.categorical_attribute.size, 1)


if __name__ == '__main__':
  absltest.main()

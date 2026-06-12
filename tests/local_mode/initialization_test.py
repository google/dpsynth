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
import numpy as np


class InitializationTest(absltest.TestCase):

  def test_numerical_initializer_dp_event(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    rng = np.random.default_rng(0)
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

    encoded_data = [measurement.transform_fn(x) for x in data]
    counts = np.bincount(encoded_data)

    # Expected Partitioning: 1 2 3 | 4 5 | 6 7 | 8 9
    np.testing.assert_array_equal(counts, [3, 2, 2, 2])


class CategoricalInitializerTest(absltest.TestCase):

  def test_dp_event(self):
    attr = domain.CategoricalAttribute(possible_values=['A', 'B', 'C'])
    rng = np.random.default_rng(0)
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


if __name__ == '__main__':
  absltest.main()

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
        name='test', num_partitions=4, attribute=attr, rng=rng
    )
    event = initializer.dp_event(1.0)
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    self.assertLen(event.events, 2)
    for e in event.events:
      self.assertIsInstance(e, dp_accounting.ExponentialMechanismDpEvent)

  def test_numerical_initializer_call(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=10)
    rng = np.random.default_rng(0)
    initializer = initialization.NumericalInitializer(
        name='test', num_partitions=4, attribute=attr, rng=rng
    )

    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Level 0 median [1..9] --> 5
    # Level 1 medians: [1..5] --> 3, [6..9] --> 7.5
    measurement = initializer(np.inf, data)

    self.assertIsInstance(measurement, initialization.ColumnMeasurement)
    self.assertEqual(measurement.categorical_attribute.size, 4)
    self.assertIsNone(measurement.measurement)

    encoded_data = [measurement.transform_fn(x) for x in data]
    counts = np.bincount(encoded_data)

    # Expected Partitioning: 1 2 3 | 4 5 | 6 7 | 8 9
    np.testing.assert_array_equal(counts, [3, 2, 2, 2])


if __name__ == '__main__':
  absltest.main()

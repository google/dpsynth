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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth.pipeline_transformations import model
import jax.numpy as jnp
import mbi
import pandas as pd
import pipeline_dp


class ModelTest(parameterized.TestCase):

  def test_fit_model(self):
    # Create test data
    domain = mbi.Domain.fromdict({'a': 2, 'b': 2})
    linear_measurements = [
        mbi.LinearMeasurement(
            noisy_measurement=jnp.array([3, 7, 2, 8]),
            clique=('a', 'b'),
        ),
        mbi.LinearMeasurement(
            noisy_measurement=jnp.array([4, 6]),
            clique=('a',),
        ),
        mbi.LinearMeasurement(
            noisy_measurement=jnp.array([2, 8]),
            clique=('b',),
        ),
    ]

    result_list = list(
        model.fit_model(
            pipeline_dp.LocalBackend(),
            [linear_measurements],
            [domain],
        )
    )
    self.assertLen(result_list, 1)
    result = result_list[0]
    self.assertIsInstance(result, mbi.MarkovRandomField)

  @parameterized.parameters(3, None)
  def test_generate_synthetic_data(self, num_records: int | None):
    mock_model = mock.create_autospec(mbi.MarkovRandomField, instance=True)
    mock_model.synthetic_data.return_value = mbi.dataset.Dataset(
        pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]}),
        domain=mbi.Domain(
            attributes=('col2', 'col3', 'col1'), shape=(8, 10, 4)
        ),
    )
    if num_records is None:
      mock_model.total = 3
    domain = mbi.Domain(attributes=('col2', 'col3', 'col1'), shape=(8, 10, 4))
    backend = pipeline_dp.LocalBackend()

    result = model.generate_synthetic_data(
        backend, [mock_model], [domain], num_records
    )
    result = list(result)

    # Assert that the output is as expected. Note that the order of the elements
    # in the tuple corresponds to the order of the attributes in the domain.
    expected_result = [(4, 7, 1), (5, 8, 2), (6, 9, 3)]
    self.assertEqual(result, expected_result)
    mock_model.synthetic_data.assert_called_once_with(rows=3)

  def test_get_num_records_per_task(self):
    self.assertEqual(model._get_num_records_per_task(0, 10), [])
    self.assertEqual(model._get_num_records_per_task(10, 100), [10])
    self.assertEqual(model._get_num_records_per_task(91, 10), [10] * 9 + [1])
    self.assertEqual(model._get_num_records_per_task(10**6, 10**6), [10**6])
    self.assertEqual(
        model._get_num_records_per_task(10**6 + 1, 10**6), [10**6, 1]
    )
    self.assertEqual(
        model._get_num_records_per_task(2 * 10**6 + 10, 10**6),
        [10**6, 10**6, 10],
    )


if __name__ == '__main__':
  absltest.main()

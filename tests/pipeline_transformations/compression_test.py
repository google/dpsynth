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
from dpsynth.pipeline_transformations import compression
import mbi
import numpy as np
import pipeline_dp


class DomainCompressionTest(absltest.TestCase):

  def test_compress_discrete_domain(self):
    backend = pipeline_dp.LocalBackend()

    compression_transforms = compression.get_domain_compression_transforms(
        [[
            mbi.LinearMeasurement(np.array([10, 1, 24, 1]), ('col1',), 1.0),
            mbi.LinearMeasurement(np.array([10, 1, 23, 6]), ('col2',), 1.0),
            mbi.LinearMeasurement(np.array([1, 2, 2, 3]), ('col3',), 1.0),
        ]],
        backend,
        'Get compression transforms',
    )
    compression_transforms_list = list(compression_transforms)[0]

    self.assertLen(compression_transforms_list, 3)
    self.assertEqual(compression_transforms_list[0][0], 3)
    self.assertEqual(compression_transforms_list[1][0], 4)
    self.assertEqual(compression_transforms_list[2][0], 2)

  def test_apply_compression_transforms(self):
    backend = pipeline_dp.LocalBackend()
    linear_measurements = [
        mbi.LinearMeasurement(np.array([23, 22, 2, 2, 10, 10]), ('col1',), 1.0),
        mbi.LinearMeasurement(np.array([1, 2, 2, 4, 5, 10]), ('col2',), 1.0),
        mbi.LinearMeasurement(np.array([1, 2, 2, 4, 6, 1]), ('col3',), 2.0),
    ]
    compression_transforms = compression.get_domain_compression_transforms(
        [linear_measurements], backend, 'Get compression transforms'
    )

    compressed_data = compression.apply_compression_transforms(
        [linear_measurements],
        compression_transforms,
        backend,
        'Compress data',
    )
    compressed_data_list = list(compressed_data)[0]

    def verify_compressed_values(actual, expected_values):
      self.assertContainsSubsequence(actual, expected_values)
      self.assertLen(actual, len(expected_values) + 1)

    self.assertLen(compressed_data_list, 3)
    self.assertEqual(compressed_data_list[0].clique, ('col1',))
    verify_compressed_values(
        compressed_data_list[0].noisy_measurement,
        [23, 22, 10, 10],
    )
    self.assertEqual(compressed_data_list[1].clique, ('col2',))
    verify_compressed_values(
        compressed_data_list[1].noisy_measurement,
        [4, 5, 10],
    )
    self.assertEqual(compressed_data_list[2].clique, ('col3',))
    verify_compressed_values(
        compressed_data_list[2].noisy_measurement,
        [6],
    )


if __name__ == '__main__':
  absltest.main()

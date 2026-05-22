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
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import independent_mechanism
import mbi
import numpy as np
import pipeline_dp


class IndependentMechanismTest(absltest.TestCase):

  def _get_dataset_descriptor(self, measurements: list[list[float]]):
    attributes = []
    for i, measurement in enumerate(measurements):
      attributes.append(
          dataset_descriptor.AttributeDescriptor(
              name=f"attr{i}",
              data_type=dataset_descriptor.DataType.INT,
              measurement=mbi.LinearMeasurement(
                  noisy_measurement=np.array(measurement),
                  clique=(i,),
                  stddev=1.0,
              ),
          )
      )
    return dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=mock.MagicMock()
    )

  def test_fit_model(self):
    backend = pipeline_dp.LocalBackend()
    descriptor = self._get_dataset_descriptor(
        [[1000, 2000, 3000], [3000, 3000]]
    )
    model = list(independent_mechanism.fit_model(backend, [descriptor]))
    self.assertLen(model, 1)
    self.assertIsInstance(model[0], mbi.MarkovRandomField)
    self.assertEqual(
        model[0].domain, mbi.Domain(attributes=(0, 1), shape=(3, 2))
    )

    data = model[0].synthetic_data(10000)
    self.assertEqual(data.records, 10000)
    counts = data.project([1]).datavector()
    self.assertBetween(counts[0], 4950, 5050)
    self.assertBetween(counts[1], 4950, 5050)

  def test_fit_model_negative_measurements(self):
    backend = pipeline_dp.LocalBackend()
    descriptor = self._get_dataset_descriptor([[10, 20, 3], [100, -50]])
    model = list(independent_mechanism.fit_model(backend, [descriptor]))
    self.assertLen(model, 1)
    self.assertIsInstance(model[0], mbi.MarkovRandomField)
    self.assertEqual(
        model[0].domain, mbi.Domain(attributes=(0, 1), shape=(3, 2))
    )


if __name__ == "__main__":
  absltest.main()

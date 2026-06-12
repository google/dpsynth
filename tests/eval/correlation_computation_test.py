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
from eval import correlation_computation
from eval import types
from dpsynth.pipeline_transformations import diagnostic_info
import pipeline_dp


class CorrelationComputationTest(absltest.TestCase):

  def test_compute_correlations(self):
    backend = pipeline_dp.LocalBackend()
    config = diagnostic_info.TabularEvalConfig(
        attributes=["attr1", "attr2", "attr3"],
        attribute_types=[
            types.DataType.STRING.value,
            types.DataType.STRING.value,
            types.DataType.DATA_TYPE_UNSPECIFIED.value,
        ],
    )
    # attr1 and attr2 are categorical, attr3 is numerical.
    # Only (attr1, attr2) correlation should be computed.
    data = [
        ("a", "x", 1.0),
        ("a", "x", 2.0),
        ("b", "y", 3.0),
        ("b", "y", 4.0),
    ]
    # Perfect correlation between attr1 and attr2

    correlations_col = correlation_computation.compute_correlations(
        data, config, backend
    )
    correlations = list(correlations_col)[0]

    self.assertLen(correlations, 1)
    corr = correlations[0]
    self.assertEqual(corr.attribute_x, "attr1")
    self.assertEqual(corr.attribute_y, "attr2")
    self.assertAlmostEqual(corr.value, 1.0)

  def test_compute_correlation_distance(self):
    backend = pipeline_dp.LocalBackend()

    report = diagnostic_info.TabularEvalReport()
    report.original_dataset_statistics.correlations.add(
        attribute_x="a", attribute_y="b", value=0.5
    )
    report.synthetic_dataset_statistics.correlations.add(
        attribute_x="a", attribute_y="b", value=0.3
    )

    report_col = [report]
    result_report_col = correlation_computation.compute_correlation_distance(
        report_col, backend
    )
    result_report = list(result_report_col)[0]

    # distance = sqrt((0.5 - 0.3)^2 / 1) = 0.2
    self.assertAlmostEqual(result_report.correlation_distance, 0.2)

  def test_compute_correlations_empty_categorical(self):
    backend = pipeline_dp.LocalBackend()
    config = diagnostic_info.TabularEvalConfig(
        attributes=["attr1", "attr2"],
        attribute_types=[
            types.DataType.DATA_TYPE_UNSPECIFIED.value,
            types.DataType.DATA_TYPE_UNSPECIFIED.value,
        ],
    )
    data = [(1.0, 2.0), (3.0, 4.0)]

    correlations_col = correlation_computation.compute_correlations(
        data, config, backend
    )
    correlations = list(correlations_col)[0]
    self.assertEmpty(correlations)


if __name__ == "__main__":
  absltest.main()

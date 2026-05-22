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
from dpsynth.pipeline_transformations import diagnostic_info
import pipeline_dp


class DiagnosticInfoTest(absltest.TestCase):

  def test_update_diagnostic_info(self):
    backend = pipeline_dp.LocalBackend()
    diag_info = diagnostic_info.DiagnosticInformation()
    diag_info.epsilon = 1.0

    diagnostic_info_collection = [diag_info]
    errors_singleton = [[((1, 2), 0.5), ((3,), 0.1)]]
    selected_marginal = [[(1, 2)]]

    result_collection = diagnostic_info.update_diagnostic_info(
        backend,
        diagnostic_info_collection,
        errors_singleton,
        selected_marginal,
        "TestStage",
    )

    result = list(result_collection)
    self.assertLen(result, 1)
    updated_diag_info = result[0]

    self.assertEqual(updated_diag_info.epsilon, 1.0)
    self.assertLen(updated_diag_info.round_info, 1)
    round_info = updated_diag_info.round_info[0]

    self.assertLen(round_info.l1_distances, 2)
    self.assertEqual(
        list(round_info.l1_distances[0].attributes.attributes), [1, 2]
    )
    self.assertAlmostEqual(round_info.l1_distances[0].value, 0.5)
    self.assertEqual(
        list(round_info.l1_distances[1].attributes.attributes), [3]
    )
    self.assertAlmostEqual(round_info.l1_distances[1].value, 0.1)

    self.assertLen(round_info.selected_attributes, 1)
    self.assertEqual(list(round_info.selected_attributes[0].attributes), [1, 2])


if __name__ == "__main__":
  absltest.main()

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
from eval import tabular_eval
from eval import types
from dpsynth.pipeline_transformations import diagnostic_info
import pipeline_dp


class TabularEvalTest(absltest.TestCase):

  def test_evaluate(self):
    backend = pipeline_dp.LocalBackend()
    original_data = [("a",), ("b",)]
    synthetic_data = [("a",), ("c",)]
    config = diagnostic_info.TabularEvalConfig(
        attributes=["cat"],
        attribute_types=[types.DataType.STRING.value],
    )

    eval_report_col = tabular_eval.evaluate(
        original_data, synthetic_data, config, backend
    )
    eval_report = list(eval_report_col)[0]

    self.assertIsNotNone(eval_report.original_dataset_statistics)
    self.assertIsNotNone(eval_report.synthetic_dataset_statistics)
    self.assertIsNotNone(eval_report.attribute_eval_reports)

    self.assertLen(eval_report.attribute_eval_reports, 1)

    cat_report = eval_report.attribute_eval_reports[0]
    self.assertEqual(cat_report.attribute_name, "cat")
    self.assertIsNotNone(cat_report.tv_distance)


if __name__ == "__main__":
  absltest.main()

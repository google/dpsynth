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
from dpsynth.eval import one_way_distribution_computation
from dpsynth.pipeline_transformations import diagnostic_info
import pipeline_dp


class OneWayDistributionComputationTest(absltest.TestCase):

  def test_compute_tv_distance(self):
    dist1 = [("a", 10), ("b", 10)]
    dist2 = [("a", 15), ("b", 5)]
    # p = [0.5, 0.5], q = [0.75, 0.25]
    # TV = 0.5 * (|0.5 - 0.75| + |0.5 - 0.25|) = 0.5 * (0.25 + 0.25) = 0.25
    self.assertAlmostEqual(
        one_way_distribution_computation._compute_tv_distance(dist1, dist2),
        0.25,
    )

  def test_compute_chi2_pvalue(self):
    sample1 = {"a": 100, "b": 100}
    sample2 = {"a": 100, "b": 100}
    p_val = one_way_distribution_computation._compute_chi2_pvalue(
        sample1, sample2
    )
    self.assertAlmostEqual(p_val, 1.0)

    sample1 = {"a": 100, "b": 0}
    sample2 = {"a": 0, "b": 100}
    p_val = one_way_distribution_computation._compute_chi2_pvalue(
        sample1, sample2
    )
    self.assertLess(p_val, 0.05)

  def test_compute_one_way_marginal_distance(self):
    backend = pipeline_dp.LocalBackend()

    original_stats = diagnostic_info.DatasetStatistics(num_records=100)
    attr1_stats = original_stats.attribute_statistics.add()
    attr1_stats.attribute_name = "attr1"
    attr1_stats.num_non_none_values = 100
    attr1_stats.categorical_statistics.num_categories = 2
    attr1_stats.categorical_statistics.category_counts.add(
        category="a", count=50
    )
    attr1_stats.categorical_statistics.category_counts.add(
        category="b", count=50
    )

    synthetic_stats = diagnostic_info.DatasetStatistics(num_records=100)
    s_attr1_stats = synthetic_stats.attribute_statistics.add()
    s_attr1_stats.attribute_name = "attr1"
    s_attr1_stats.num_non_none_values = 100
    s_attr1_stats.categorical_statistics.num_categories = 2
    s_attr1_stats.categorical_statistics.category_counts.add(
        category="a", count=60
    )
    s_attr1_stats.categorical_statistics.category_counts.add(
        category="b", count=40
    )

    eval_report = diagnostic_info.TabularEvalReport()
    eval_report.original_dataset_statistics.CopyFrom(original_stats)
    eval_report.synthetic_dataset_statistics.CopyFrom(synthetic_stats)

    eval_report_col = [eval_report]
    result_eval_report = list(
        one_way_distribution_computation.compute_one_way_marginal_distance(
            eval_report_col, backend
        )
    )[0]

    self.assertIsNotNone(result_eval_report.attribute_eval_reports)
    self.assertLen(result_eval_report.attribute_eval_reports, 1)
    report = result_eval_report.attribute_eval_reports[0]
    self.assertEqual(report.attribute_name, "attr1")
    # TV = 0.5 * (|0.5 - 0.6| + |0.5 - 0.4|) = 0.5 * (0.1 + 0.1) = 0.1
    self.assertAlmostEqual(report.tv_distance, 0.1)
    self.assertGreater(report.chi2_pvalue, 0)


if __name__ == "__main__":
  absltest.main()

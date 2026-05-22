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
from dpsynth.eval import attribute_statistics
from dpsynth.eval import types
from dpsynth.pipeline_transformations import diagnostic_info
import pipeline_dp


class AttributeStatisticsTest(absltest.TestCase):

  def test_compute_categorical_statistics(self):
    backend = pipeline_dp.LocalBackend()
    data = [(1.0, "a"), (2.0, "b"), (3.0, "a"), (None, "c")]
    config = diagnostic_info.TabularEvalConfig(
        attributes=["num", "cat"],
        attribute_types=[
            types.DataType.INT_CATEGORICAL.value,
            types.DataType.STRING.value,
        ],
    )

    dataset_statistics_col, _ = attribute_statistics.compute_dataset_statistics(
        data, config, backend
    )
    dataset_statistics = list(dataset_statistics_col)[0]

    # Categorical attribute "cat" at index 1
    cat_stats = dataset_statistics.attribute_statistics[1]
    self.assertEqual(cat_stats.attribute_name, "cat")
    self.assertEqual(cat_stats.num_non_none_values, 4)
    self.assertIsNotNone(cat_stats.categorical_statistics)
    self.assertEqual(cat_stats.categorical_statistics.num_categories, 3)
    counts = {
        c.category: c.count
        for c in cat_stats.categorical_statistics.category_counts
    }
    self.assertEqual(counts["a"], 2)
    self.assertEqual(counts["b"], 1)
    self.assertEqual(counts["c"], 1)


if __name__ == "__main__":
  absltest.main()

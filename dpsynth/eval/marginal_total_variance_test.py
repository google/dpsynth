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
from dpsynth.eval import marginal_total_variance
import pipeline_dp


class MarginalTotalVarianceTest(absltest.TestCase):

  def test_compute_multi_way_marginal_distance(self):
    backend = pipeline_dp.LocalBackend()

    # Two categorical attributes: 0 and 1
    # Original data:
    # (A, X) - 40
    # (A, Y) - 10
    # (B, X) - 10
    # (B, Y) - 40
    original_data = (
        [("A", "X")] * 40
        + [("A", "Y")] * 10
        + [("B", "X")] * 10
        + [("B", "Y")] * 40
    )

    # Synthetic data:
    # (A, X) - 25
    # (A, Y) - 25
    # (B, X) - 25
    # (B, Y) - 25
    synthetic_data = (
        [("A", "X")] * 25
        + [("A", "Y")] * 25
        + [("B", "X")] * 25
        + [("B", "Y")] * 25
    )

    categorical_indices = [0, 1]
    original_size = [len(original_data)]
    synthetic_size = [len(synthetic_data)]

    marginals_col = marginal_total_variance.compute_multi_way_marginal_distance(
        original_data,
        synthetic_data,
        categorical_indices,
        original_size,
        synthetic_size,
        backend,
    )
    marginals = list(marginals_col)

    # Combinations expected: (0,), (1,), (0, 1)
    self.assertLen(marginals, 3)

    marginals_dict = {
        tuple(m.attribute_indices): m.tv_distance for m in marginals
    }

    # TV(1-way, index 0):
    # Org: A=50/100, B=50/100 -> [0.5, 0.5]
    # Syn: A=50/100, B=50/100 -> [0.5, 0.5]
    # TV = 0.5 * (|0.5-0.5| + |0.5-0.5|) = 0
    self.assertAlmostEqual(marginals_dict[(0,)], 0.0)

    # TV(1-way, index 1):
    # Org: X=50/100, Y=50/100 -> [0.5, 0.5]
    # Syn: X=50/100, Y=50/100 -> [0.5, 0.5]
    # TV = 0
    self.assertAlmostEqual(marginals_dict[(1,)], 0.0)

    # TV(2-way, indices (0, 1)):
    # Org: (A,X)=0.4, (A,Y)=0.1, (B,X)=0.1, (B,Y)=0.4
    # Syn: (A,X)=0.25, (A,Y)=0.25, (B,X)=0.25, (B,Y)=0.25
    # TV = 0.5 * (|0.4-0.25| + |0.1-0.25| + |0.1-0.25| + |0.4-0.25|)
    #    = 0.5 * (0.15 + 0.15 + 0.15 + 0.15) = 0.5 * 0.6 = 0.3
    self.assertAlmostEqual(marginals_dict[(0, 1)], 0.3)

  def test_compute_multi_way_marginal_distance_with_unseen(self):
    backend = pipeline_dp.LocalBackend()

    # Original data:
    # (A, X) - 50
    # (B, Y) - 50
    original_data = [("A", "X")] * 50 + [("B", "Y")] * 50

    # Synthetic data:
    # (A, X) - 40
    # (A, Y) - 20  <-- Unseen!
    # (B, Y) - 40
    synthetic_data = [("A", "X")] * 40 + [("A", "Y")] * 20 + [("B", "Y")] * 40

    categorical_indices = [0, 1]
    original_size = [len(original_data)]
    synthetic_size = [len(synthetic_data)]

    marginals_col = marginal_total_variance.compute_multi_way_marginal_distance(
        original_data,
        synthetic_data,
        categorical_indices,
        original_size,
        synthetic_size,
        backend,
    )
    marginals = list(marginals_col)

    # Combinations expected: (0,), (1,), (0, 1)
    self.assertLen(marginals, 3)
    marginals_dict = {tuple(m.attribute_indices): m for m in marginals}

    # 1-way should have no unseen
    m_0 = marginals_dict[(0,)]
    self.assertEqual(m_0.num_unseen_combinations, 0)
    self.assertEqual(m_0.unseen_occurrences, 0)
    self.assertEmpty(m_0.top_unseen_combinations)

    # 2-way should have 1 unseen combination: (A, Y)
    m_0_1 = marginals_dict[(0, 1)]
    self.assertEqual(m_0_1.num_unseen_combinations, 1)
    self.assertEqual(m_0_1.unseen_occurrences, 20)
    self.assertLen(m_0_1.top_unseen_combinations, 1)

    top_unseen = m_0_1.top_unseen_combinations[0]
    self.assertEqual(top_unseen.count, 20)
    self.assertEqual(list(top_unseen.values), ["A", "Y"])


if __name__ == "__main__":
  absltest.main()

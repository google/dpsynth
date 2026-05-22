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

"""Tests for quantiles primitives."""

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth.local_mode import primitives
import numpy as np


class QuantilesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_median_basic(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    med = primitives.median(self.rng, data, lower=0.0, upper=10.0, zcdp_rho=100)
    self.assertBetween(med, 2.0, 4.5)

  def test_median_empty(self):
    data = np.array([])
    med = primitives.median(self.rng, data, lower=0.0, upper=10.0, zcdp_rho=1.0)
    self.assertBetween(med, 0.0, 10.0)

  def test_quantiles_basic(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    qs = primitives.quantiles(
        self.rng, data, lower=0.0, upper=10.0, num_partitions=4, zcdp_rho=100.0
    )
    self.assertLen(qs, 3)
    self.assertTrue(all(qs[i] <= qs[i + 1] for i in range(len(qs) - 1)))

  def test_quantiles_empty(self):
    data = np.array([])
    qs = primitives.quantiles(
        self.rng, data, lower=0.0, upper=10.0, num_partitions=4, zcdp_rho=1.0
    )
    self.assertLen(qs, 3)
    self.assertTrue(all(qs[i] <= qs[i + 1] for i in range(len(qs) - 1)))

  def test_median_with_duplicates(self):
    data = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 6.0])
    med = primitives.median(self.rng, data, lower=0.0, upper=10.0, zcdp_rho=100)
    self.assertBetween(med, 1.5, 2.5)

  def test_median_with_out_of_bounds(self):
    data = np.array([-5.0, -2.0, 1.0, 2.0, 3.0, 12.0, 15.0])
    med = primitives.median(self.rng, data, lower=0.0, upper=10.0, zcdp_rho=100)
    self.assertBetween(med, 1.0, 3.0)

  def test_quantiles_with_duplicates_and_clamping(self):
    data = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 12.0])
    qs = primitives.quantiles(
        self.rng, data, lower=0.0, upper=10.0, num_partitions=4, zcdp_rho=100
    )
    self.assertLen(qs, 3)
    self.assertTrue(all(qs[i] <= qs[i + 1] for i in range(len(qs) - 1)))

  def test_median_zcdp_rho_zero(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    med = primitives.median(self.rng, data, lower=0.0, upper=10.0, zcdp_rho=0.0)
    self.assertBetween(med, 0.0, 10.0)

  def test_median_zcdp_rho_inf(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    med = primitives.median(self.rng, data, 0, 10, zcdp_rho=np.inf)
    self.assertEqual(med, 3.0)

  def test_quantiles_zcdp_rho_zero(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    qs = primitives.quantiles(
        self.rng, data, lower=0.0, upper=10.0, num_partitions=4, zcdp_rho=0.0
    )
    self.assertLen(qs, 3)
    self.assertTrue(all(0.0 <= q <= 10.0 for q in qs))

  def test_quantiles_zcdp_rho_inf(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    qs = primitives.quantiles(
        self.rng, data, lower=0.0, upper=10.0, num_partitions=4, zcdp_rho=np.inf
    )
    self.assertLen(qs, 3)
    self.assertEqual(qs, [2.5, 4.0, 6.0])


class SelectPartitionsSipsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_basic_operation(self):
    data = np.array([1] * 50 + [2] * 5)
    selected, counts, sigma = primitives.select_partitions_sips(
        self.rng, data, gdp_budget=10.0, delta=1e-5
    )
    self.assertIn(1, selected)
    self.assertEqual(sigma, 1.0 / np.sqrt(10.0))
    self.assertEqual(selected.size, counts.size)

  def test_empty_data(self):
    data = np.array([], dtype=int)
    selected, counts, sigma = primitives.select_partitions_sips(
        self.rng, data, gdp_budget=1.0, delta=1e-5
    )
    self.assertEmpty(selected)
    self.assertEmpty(counts)
    self.assertEqual(sigma, 1.0)

  def test_infinite_budget(self):
    data = np.array([1, 2, 3, 4, 5])
    selected, counts, sigma = primitives.select_partitions_sips(
        self.rng, data, gdp_budget=np.inf, delta=0.1
    )
    self.assertCountEqual(selected, [1, 2, 3, 4, 5])
    self.assertEqual(sigma, 0.0)
    np.testing.assert_array_equal(counts, np.ones(5))

  def test_zero_budget_raises(self):
    data = np.array([1, 2, 3])
    with self.assertRaises(ValueError):
      primitives.select_partitions_sips(
          self.rng, data, gdp_budget=-0.1, delta=1e-5
      )
    with self.assertRaises(ValueError):
      primitives.select_partitions_sips(
          self.rng, data, gdp_budget=1.0, delta=-0.001
      )

  def test_string_data_type(self):
    data = np.array(["a", "b", "a", "c"])
    selected, _, _ = primitives.select_partitions_sips(
        self.rng, data, gdp_budget=10.0, delta=1e-5
    )
    self.assertTrue(all(isinstance(p, str) for p in selected))

  def test_user_level_dp_weighting(self):
    data = np.array([1] * 10 + [2])
    user_ids = np.array([1] * 10 + [2])
    selected, counts, sigma = primitives.select_partitions_sips(
        self.rng, data, gdp_budget=100.0, delta=1e-5, user_ids=user_ids
    )
    self.assertIn(1, selected)
    self.assertIn(2, selected)
    for c in counts:
      self.assertAlmostEqual(c, 1.0, delta=3 * sigma)

  @parameterized.named_parameters(
      ("item_level_default_rounds", None, None),
      ("item_level_3_rounds", None, 3),
      ("user_level_default_rounds", np.array([1, 2, 3]), None),
      ("user_level_5_rounds", np.array([1, 2, 3]), 5),
  )
  def test_configurations(self, user_ids, num_rounds):
    data = np.array([1, 2, 3])
    gdp_budget = 10.0
    (
        _,
        _,
    ) = sigma = primitives.select_partitions_sips(
        self.rng,
        data,
        gdp_budget=gdp_budget,
        delta=1e-5,
        num_rounds=num_rounds,
        user_ids=user_ids,
    )
    self.assertLessEqual(sigma, 1.0 / np.sqrt(gdp_budget))

  def test_mismatched_user_ids_raises(self):
    data = np.array([1, 2, 3])
    user_ids = np.array([1, 2])
    with self.assertRaises(ValueError):
      primitives.select_partitions_sips(
          self.rng, data, gdp_budget=10.0, delta=1e-5, user_ids=user_ids
      )

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

import unittest

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth.local_mode import primitives
import numpy as np


@unittest.skip(
    "SIPS tests are broken at HEAD; will be replaced by Gaussian partition"
    " selection."
)
class SelectPartitionsSipsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_basic_operation(self):
    data = np.array([1] * 50 + [2] * 5)
    selected, counts, sigma = primitives._select_partitions_sips(
        self.rng, data, gdp_budget=10.0, delta=1e-5
    )
    self.assertIn(1, selected)
    self.assertEqual(sigma, 1.0 / np.sqrt(10.0))
    self.assertEqual(selected.size, counts.size)

  def test_empty_data(self):
    data = np.array([], dtype=int)
    selected, counts, sigma = primitives._select_partitions_sips(
        self.rng, data, gdp_budget=1.0, delta=1e-5
    )
    self.assertEmpty(selected)
    self.assertEmpty(counts)
    self.assertEqual(sigma, 1.0)

  def test_infinite_budget(self):
    data = np.array([1, 2, 3, 4, 5])
    selected, counts, sigma = primitives._select_partitions_sips(
        self.rng, data, gdp_budget=np.inf, delta=0.1
    )
    self.assertCountEqual(selected, [1, 2, 3, 4, 5])
    self.assertEqual(sigma, 0.0)
    np.testing.assert_array_equal(counts, np.ones(5))

  def test_zero_budget_raises(self):
    data = np.array([1, 2, 3])
    with self.assertRaises(ValueError):
      primitives._select_partitions_sips(
          self.rng, data, gdp_budget=-0.1, delta=1e-5
      )
    with self.assertRaises(ValueError):
      primitives._select_partitions_sips(
          self.rng, data, gdp_budget=1.0, delta=-0.001
      )

  def test_string_data_type(self):
    data = np.array(["a", "b", "a", "c"])
    selected, _, _ = primitives._select_partitions_sips(
        self.rng, data, gdp_budget=10.0, delta=1e-5
    )
    self.assertTrue(all(isinstance(p, str) for p in selected))

  def test_user_level_dp_weighting(self):
    # Partition 1 has 10 unique users (1 to 10), each contributing 1 time.
    # Partition 2 has 1 user (11) contributing 10 times.
    data = np.array([1] * 10 + [2] * 10)
    user_ids = np.array(list(range(1, 11)) + [11] * 10)

    selected, _, _ = primitives._select_partitions_sips(
        self.rng, data, gdp_budget=10.0, delta=1e-5, user_ids=user_ids
    )
    self.assertIn(1, selected)
    self.assertNotIn(2, selected)

  @parameterized.named_parameters(
      ("item_level_default_rounds", None, None),
      ("item_level_3_rounds", None, 3),
      ("user_level_default_rounds", np.array([1, 2, 3]), None),
      ("user_level_5_rounds", np.array([1, 2, 3]), 5),
  )
  def test_configurations(self, user_ids, num_rounds):
    data = np.array([1, 2, 3])
    gdp_budget = 10.0
    _, _, sigma = primitives._select_partitions_sips(
        self.rng,
        data,
        gdp_budget=gdp_budget,
        delta=1e-5,
        num_rounds=num_rounds,
        user_ids=user_ids,
    )
    # Calculate expected max_sigma based on budget allocation
    if num_rounds is None:
      num_rounds = 1 if user_ids is None else 3
    allocation_factor = 0.3  # default in primitives.py
    fractions = allocation_factor ** np.arange(num_rounds)[::-1]
    fractions /= fractions.sum()
    gdp_rounds = gdp_budget * fractions
    expected_max_sigma = float(np.max(1.0 / np.sqrt(gdp_rounds)))

    self.assertAlmostEqual(sigma, expected_max_sigma)

  def test_mismatched_user_ids_raises(self):
    data = np.array([1, 2, 3])
    user_ids = np.array([1, 2])
    with self.assertRaises(ValueError):
      primitives._select_partitions_sips(
          self.rng, data, gdp_budget=10.0, delta=1e-5, user_ids=user_ids
      )


class SelectPartitionsGaussianThresholdingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_basic_operation(self):
    data = np.array([1] * 50 + [2] * 5)
    selected_partitions, estimated_counts, _ = (
        primitives.select_partitions_gaussian_thresholding(
            self.rng, data, gdp_budget=10.0, delta=1e-5
        )
    )
    self.assertIn(1, selected_partitions)
    self.assertEqual(selected_partitions.size, estimated_counts.size)

  def test_empty_data(self):
    data = np.array([], dtype=int)
    selected_partitions, estimated_counts, _ = (
        primitives.select_partitions_gaussian_thresholding(
            self.rng, data, gdp_budget=1.0, delta=1e-5
        )
    )
    self.assertEmpty(selected_partitions)
    self.assertEmpty(estimated_counts)

  def test_high_budget_selects_all(self):
    data = np.array([1, 2, 3, 4, 5])
    selected_partitions, _, _ = (
        primitives.select_partitions_gaussian_thresholding(
            self.rng, data, gdp_budget=np.inf, delta=0.1
        )
    )
    self.assertCountEqual(selected_partitions, [1, 2, 3, 4, 5])

  def test_rare_items_not_selected(self):
    # One item with many occurrences, another with just 1.
    # With moderate budget and tight delta, the rare item should be dropped.
    data = np.array([1] * 100 + [2])
    selected_partitions, _, _ = (
        primitives.select_partitions_gaussian_thresholding(
            self.rng, data, gdp_budget=0.5, delta=1e-6
        )
    )
    self.assertIn(1, selected_partitions)
    self.assertNotIn(2, selected_partitions)

  def test_string_data_type(self):
    data = np.array(["a", "b", "a", "a", "c", "a", "c"])
    selected_partitions, _, _ = (
        primitives.select_partitions_gaussian_thresholding(
            self.rng, data, gdp_budget=10.0, delta=1e-5
        )
    )
    self.assertTrue(all(isinstance(p, str) for p in selected_partitions))

  def test_min_count_filters_low_count_partitions(self):
    # Partition 1 has count 50, partition 2 has count 3.
    data = np.array([1] * 50 + [2] * 3)
    selected, _, _ = primitives.select_partitions_gaussian_thresholding(
        self.rng, data, gdp_budget=10.0, delta=1e-5, min_count=5
    )
    self.assertIn(1, selected)
    self.assertNotIn(2, selected)

  def test_min_count_one_matches_default(self):
    data = np.array([1] * 50 + [2] * 5)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    result1 = primitives.select_partitions_gaussian_thresholding(
        rng1, data, gdp_budget=10.0, delta=1e-5
    )
    result2 = primitives.select_partitions_gaussian_thresholding(
        rng2, data, gdp_budget=10.0, delta=1e-5, min_count=1
    )
    np.testing.assert_array_equal(result1[0], result2[0])
    np.testing.assert_array_equal(result1[1], result2[1])

  def test_min_count_all_filtered_returns_empty(self):
    data = np.array([1, 2, 3])
    selected, counts, _ = primitives.select_partitions_gaussian_thresholding(
        self.rng, data, gdp_budget=10.0, delta=1e-5, min_count=5
    )
    self.assertEmpty(selected)
    self.assertEmpty(counts)

  def test_min_count_zero_raises(self):
    data = np.array([1, 2, 3])
    with self.assertRaises(ValueError):
      primitives.select_partitions_gaussian_thresholding(
          self.rng, data, gdp_budget=1.0, delta=1e-5, min_count=0
      )

  def test_min_count_increases_threshold(self):
    # With very high budget (no noise), threshold is approximately min_count.
    # Partitions with count exactly at min_count should pass.
    data = np.array([1] * 10 + [2] * 10)
    selected, _, _ = primitives.select_partitions_gaussian_thresholding(
        self.rng, data, gdp_budget=np.inf, delta=0.1, min_count=10
    )
    self.assertCountEqual(selected, [1, 2])


class EnsurePublicPartitionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_missing_partitions_appended_and_sorted(self):
    selected = np.array(["a", "c"])
    counts = np.array([10.0, 20.0])
    public = np.array(["b", "c"])
    sel, cts = primitives.ensure_public_partitions(
        self.rng, selected, counts, 0.0, public
    )
    np.testing.assert_array_equal(sel, ["a", "b", "c"])
    self.assertEqual(cts[0], 10.0)  # count for 'a'
    self.assertEqual(cts[1], 0.0)  # noise for 'b' (stddev=0)
    self.assertEqual(cts[2], 20.0)  # count for 'c'

  def test_all_present_is_noop(self):
    selected = np.array(["a", "b"])
    counts = np.array([10.0, 20.0])
    public = np.array(["a", "b"])
    sel, cts = primitives.ensure_public_partitions(
        self.rng, selected, counts, 1.0, public
    )
    np.testing.assert_array_equal(sel, selected)
    np.testing.assert_array_equal(cts, counts)

  def test_empty_selected(self):
    selected = np.array([], dtype=str)
    counts = np.array([], dtype=float)
    public = np.array(["y", "x"])
    sel, cts = primitives.ensure_public_partitions(
        self.rng, selected, counts, 1.0, public
    )
    np.testing.assert_array_equal(sel, ["x", "y"])
    self.assertLen(cts, 2)


class AddGaussianNoiseTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_scalar(self):
    noisy = primitives.add_gaussian_noise(self.rng, 100, sigma=1.0)
    self.assertIsInstance(noisy, float)
    self.assertAlmostEqual(noisy, 100.0, delta=5.0)

  def test_1d_array(self):
    counts = np.array([10, 20, 30])
    noisy = primitives.add_gaussian_noise(self.rng, counts, sigma=1.0)
    self.assertEqual(noisy.shape, (3,))
    np.testing.assert_allclose(noisy, counts, atol=5.0)

  def test_2d_array(self):
    counts = np.ones((2, 2)) * 10
    noisy = primitives.add_gaussian_noise(self.rng, counts, sigma=1.0)
    self.assertEqual(noisy.shape, (2, 2))
    np.testing.assert_allclose(noisy, counts, atol=5.0)




if __name__ == "__main__":
  absltest.main()

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
import dp_accounting
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
    mech = primitives.DPPartitionSelection(
        delta=1e-5, sigma=1.0 / np.sqrt(10.0)
    )
    result = mech(self.rng, data)
    self.assertIn(1, result.selected_partitions)
    self.assertEqual(
        result.selected_partitions.size, result.estimated_counts.size
    )

  def test_empty_data(self):
    data = np.array([], dtype=int)
    mech = primitives.DPPartitionSelection(delta=1e-5, sigma=1.0)
    result = mech(self.rng, data)
    self.assertEmpty(result.selected_partitions)
    self.assertEmpty(result.estimated_counts)

  def test_high_budget_selects_all(self):
    data = np.array([1, 2, 3, 4, 5])
    mech = primitives.DPPartitionSelection(delta=0.1, sigma=0.0)
    result = mech(self.rng, data)
    self.assertCountEqual(result.selected_partitions, [1, 2, 3, 4, 5])

  def test_rare_items_not_selected(self):
    # One item with many occurrences, another with just 1.
    # With moderate budget and tight delta, the rare item should be dropped.
    data = np.array([1] * 100 + [2])
    mech = primitives.DPPartitionSelection(delta=1e-6, sigma=1.0 / np.sqrt(0.5))
    result = mech(self.rng, data)
    self.assertIn(1, result.selected_partitions)
    self.assertNotIn(2, result.selected_partitions)

  def test_string_data_type(self):
    data = np.array(["a", "b", "a", "a", "c", "a", "c"])
    mech = primitives.DPPartitionSelection(
        delta=1e-5, sigma=1.0 / np.sqrt(10.0)
    )
    result = mech(self.rng, data)
    self.assertTrue(all(isinstance(p, str) for p in result.selected_partitions))

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


class GaussianHistogramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_basic_operation(self):
    counts = np.array([2, 3, 1, 0])
    mech = primitives.DPGaussianHistogram(domain_size=4, sigma=1.0)
    result = mech(self.rng, counts)
    self.assertLen(result.counts, 4)
    # Noisy counts should be close to true counts [2, 3, 1, 0].
    np.testing.assert_allclose(result.counts, [2, 3, 1, 0], atol=5.0)

  def test_zero_sigma(self):
    counts = np.array([2, 1, 3])
    mech = primitives.DPGaussianHistogram(domain_size=3, sigma=0.0)
    result = mech(self.rng, counts)
    np.testing.assert_array_equal(result.counts, [2, 1, 3])

  def test_empty_data(self):
    counts = np.array([0, 0, 0])
    mech = primitives.DPGaussianHistogram(domain_size=3, sigma=1.0)
    result = mech(self.rng, counts)
    self.assertLen(result.counts, 3)


# ---------------------------------------------------------------------------
# DPMechanism wrapper tests


class DPGaussianHistogramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_calibrate_and_call(self):
    mech = primitives.DPGaussianHistogram(domain_size=4)
    calibrated = mech.calibrate(zcdp_rho=0.5)
    counts = np.array([2, 3, 1, 0])
    result = calibrated(self.rng, counts)
    self.assertLen(result.counts, 4)
    np.testing.assert_allclose(result.counts, [2, 3, 1, 0], atol=5.0)

  def test_direct_sigma(self):
    mech = primitives.DPGaussianHistogram(domain_size=3, sigma=0.0)
    counts = np.array([2, 1, 3])
    np.testing.assert_array_equal(mech(self.rng, counts).counts, [2, 1, 3])

  def test_dp_event_raises_before_calibration(self):
    mech = primitives.DPGaussianHistogram(domain_size=4)
    with self.assertRaises(ValueError):
      _ = mech.dp_event

  def test_call_raises_before_calibration(self):
    mech = primitives.DPGaussianHistogram(domain_size=4)
    with self.assertRaises(ValueError):
      mech(self.rng, np.array([0, 0, 1, 0]))

  def test_dp_event_type(self):
    mech = primitives.DPGaussianHistogram(domain_size=4).calibrate(zcdp_rho=0.5)
    event = mech.dp_event
    self.assertIsInstance(event, dp_accounting.GaussianDpEvent)
    self.assertAlmostEqual(event.noise_multiplier, 1.0)


class DPGaussianCountTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_calibrate_and_call(self):
    mech = primitives.DPGaussianCount()
    calibrated = mech.calibrate(zcdp_rho=0.5)
    data = np.array([1, 2, 3, 4, 5])
    result = calibrated(self.rng, data)
    self.assertIsInstance(result, float)
    np.testing.assert_allclose(result, 5.0, atol=5.0)

  def test_zero_sigma_returns_exact_count(self):
    mech = primitives.DPGaussianCount(sigma=0.0)
    data = np.array([10, 20, 30])
    self.assertEqual(mech(self.rng, data), 3.0)

  def test_dp_event_raises_before_calibration(self):
    mech = primitives.DPGaussianCount()
    with self.assertRaises(ValueError):
      _ = mech.dp_event

  def test_dp_event_type(self):
    mech = primitives.DPGaussianCount().calibrate(zcdp_rho=0.5)
    event = mech.dp_event
    self.assertIsInstance(event, dp_accounting.GaussianDpEvent)
    self.assertAlmostEqual(event.noise_multiplier, 1.0)


if __name__ == "__main__":
  absltest.main()

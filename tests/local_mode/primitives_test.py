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


class QuantilesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_median_basic(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    med = primitives._median(self.rng, data, lower=0.0, upper=10.0, epsilon=100)
    self.assertBetween(med, 2.0, 4.5)

  def test_median_empty(self):
    data = np.array([])
    med = primitives._median(self.rng, data, lower=0.0, upper=10.0, epsilon=1.0)
    self.assertBetween(med, 0.0, 10.0)

  def test_quantiles_basic(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    eps_levels = primitives._quantile_epsilon_levels(100.0, 2)
    qs = primitives._quantiles(
        self.rng, data, lower=0.0, upper=10.0, epsilon_levels=eps_levels
    )
    self.assertLen(qs, 3)
    self.assertTrue(all(qs[i] <= qs[i + 1] for i in range(len(qs) - 1)))

  def test_quantiles_empty(self):
    data = np.array([])
    eps_levels = primitives._quantile_epsilon_levels(1.0, 2)
    qs = primitives._quantiles(
        self.rng, data, lower=0.0, upper=10.0, epsilon_levels=eps_levels
    )
    self.assertLen(qs, 3)
    self.assertTrue(all(qs[i] <= qs[i + 1] for i in range(len(qs) - 1)))

  def test_median_with_duplicates(self):
    data = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 6.0])
    med = primitives._median(self.rng, data, lower=0.0, upper=10.0, epsilon=100)
    self.assertBetween(med, 1.5, 2.5)

  def test_median_with_out_of_bounds(self):
    data = np.array([-5.0, -2.0, 1.0, 2.0, 3.0, 12.0, 15.0])
    med = primitives._median(self.rng, data, lower=0.0, upper=10.0, epsilon=100)
    self.assertBetween(med, 1.0, 3.0)

  def test_quantiles_with_duplicates_and_clamping(self):
    data = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 12.0])
    eps_levels = primitives._quantile_epsilon_levels(100, 2)
    qs = primitives._quantiles(
        self.rng, data, lower=0.0, upper=10.0, epsilon_levels=eps_levels
    )
    self.assertLen(qs, 3)
    self.assertTrue(all(qs[i] <= qs[i + 1] for i in range(len(qs) - 1)))

  def test_median_zcdp_rho_zero(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    med = primitives._median(self.rng, data, lower=0.0, upper=10.0, epsilon=0.0)
    self.assertBetween(med, 0.0, 10.0)

  def test_median_zcdp_rho_inf(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    med = primitives._median(self.rng, data, 0, 10, epsilon=np.inf)
    self.assertEqual(med, 3.0)

  def test_quantiles_zcdp_rho_zero(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    eps_levels = primitives._quantile_epsilon_levels(0.0, 2)
    qs = primitives._quantiles(
        self.rng, data, lower=0.0, upper=10.0, epsilon_levels=eps_levels
    )
    self.assertLen(qs, 3)
    self.assertTrue(all(0.0 <= q <= 10.0 for q in qs))

  def test_quantiles_zcdp_rho_inf(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    eps_levels = primitives._quantile_epsilon_levels(np.inf, 2)
    qs = primitives._quantiles(
        self.rng, data, lower=0.0, upper=10.0, epsilon_levels=eps_levels
    )
    self.assertLen(qs, 3)
    self.assertEqual(qs, [2.5, 4.0, 6.0])


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
    selected, counts, sigma = (
        primitives.select_partitions_gaussian_thresholding(
            self.rng, data, gdp_budget=10.0, delta=1e-5
        )
    )
    self.assertIn(1, selected)
    self.assertEqual(sigma, 1.0 / np.sqrt(10.0))
    self.assertEqual(selected.size, counts.size)

  def test_empty_data(self):
    data = np.array([], dtype=int)
    selected, counts, sigma = (
        primitives.select_partitions_gaussian_thresholding(
            self.rng, data, gdp_budget=1.0, delta=1e-5
        )
    )
    self.assertEmpty(selected)
    self.assertEmpty(counts)
    self.assertEqual(sigma, 1.0)

  def test_high_budget_selects_all(self):
    data = np.array([1, 2, 3, 4, 5])
    selected, _, _ = primitives.select_partitions_gaussian_thresholding(
        self.rng, data, gdp_budget=np.inf, delta=0.1
    )
    self.assertCountEqual(selected, [1, 2, 3, 4, 5])

  def test_zero_budget_raises(self):
    data = np.array([1, 2, 3])
    with self.assertRaises(ValueError):
      primitives.select_partitions_gaussian_thresholding(
          self.rng, data, gdp_budget=-0.1, delta=1e-5
      )
    with self.assertRaises(ValueError):
      primitives.select_partitions_gaussian_thresholding(
          self.rng, data, gdp_budget=1.0, delta=-0.001
      )

  def test_rare_items_not_selected(self):
    # One item with many occurrences, another with just 1.
    # With moderate budget and tight delta, the rare item should be dropped.
    data = np.array([1] * 100 + [2])
    selected, _, _ = primitives.select_partitions_gaussian_thresholding(
        self.rng, data, gdp_budget=0.5, delta=1e-6
    )
    self.assertIn(1, selected)
    self.assertNotIn(2, selected)

  def test_string_data_type(self):
    data = np.array(["a", "b", "a", "a", "c", "a", "c"])
    selected, _, _ = primitives.select_partitions_gaussian_thresholding(
        self.rng, data, gdp_budget=10.0, delta=1e-5
    )
    self.assertTrue(all(isinstance(p, str) for p in selected))


class GaussianHistogramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_basic_operation(self):
    data = np.array([0, 0, 1, 1, 1, 2])
    result = primitives._gaussian_histogram(self.rng, data, 4, sigma=1.0)
    self.assertLen(result, 4)
    # Noisy counts should be close to true counts [2, 3, 1, 0].
    np.testing.assert_allclose(result, [2, 3, 1, 0], atol=5.0)

  def test_zero_sigma(self):
    data = np.array([0, 0, 1, 2, 2, 2])
    result = primitives._gaussian_histogram(self.rng, data, 3, sigma=0.0)
    np.testing.assert_array_equal(result, [2, 1, 3])

  def test_empty_data(self):
    data = np.array([], dtype=int)
    result = primitives._gaussian_histogram(self.rng, data, 3, sigma=1.0)
    self.assertLen(result, 3)


# ---------------------------------------------------------------------------
# DPMechanism wrapper tests
# ---------------------------------------------------------------------------


class DPQuantilesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_calibrate_and_call(self):
    mech = primitives.DPQuantiles(lower=0.0, upper=10.0, num_partitions=4)
    calibrated = mech.calibrate(zcdp_rho=100.0)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = calibrated(self.rng, data)
    self.assertLen(result.quantiles, 3)

  @parameterized.parameters([0.3, 1.0, 2.718])
  def test_calibrate_default_ratio(self, zcdp_rho):
    mech = primitives.DPQuantiles(lower=0.0, upper=10.0, num_partitions=4)
    calibrated = mech.calibrate(zcdp_rho=zcdp_rho)
    # Default ratio=2: deeper level gets double the epsilon.
    events = calibrated.dp_event.events
    self.assertLen(events, 2)
    np.testing.assert_allclose(events[0].epsilon / events[1].epsilon, 2.0)

  def test_calibrate_custom_ratio(self):
    mech = primitives.DPQuantiles(lower=0.0, upper=10.0, num_partitions=4)
    calibrated = mech.calibrate(zcdp_rho=1.0, epsilon_ratio=1.0)
    # Ratio=1 means uniform epsilon across levels.
    events = calibrated.dp_event.events
    np.testing.assert_allclose(events[0].epsilon, events[1].epsilon)

  def test_dp_event_raises_before_calibration(self):
    mech = primitives.DPQuantiles(lower=0.0, upper=10.0, num_partitions=4)
    with self.assertRaises(ValueError):
      _ = mech.dp_event

  def test_dp_event_type(self):
    mech = primitives.DPQuantiles(
        lower=0.0, upper=10.0, num_partitions=4
    ).calibrate(zcdp_rho=1.0)
    event = mech.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    self.assertLen(event.events, 2)  # log2(4) = 2 levels
    for e in event.events:
      self.assertIsInstance(e, dp_accounting.ExponentialMechanismDpEvent)

  def test_dp_event_single_partition(self):
    mech = primitives.DPQuantiles(
        lower=0.0, upper=10.0, num_partitions=1
    ).calibrate(zcdp_rho=1.0)
    event = mech.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    self.assertEmpty(event.events)


class DPGaussianHistogramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_calibrate_and_call(self):
    mech = primitives.DPGaussianHistogram(domain_size=4)
    calibrated = mech.calibrate(zcdp_rho=0.5)
    data = np.array([0, 0, 1, 1, 1, 2])
    result = calibrated(self.rng, data)
    self.assertLen(result.counts, 4)
    np.testing.assert_allclose(result.counts, [2, 3, 1, 0], atol=5.0)

  def test_direct_sigma(self):
    mech = primitives.DPGaussianHistogram(domain_size=3, sigma=0.0)
    data = np.array([0, 0, 1, 2, 2, 2])
    np.testing.assert_array_equal(mech(self.rng, data).counts, [2, 1, 3])

  def test_dp_event_raises_before_calibration(self):
    mech = primitives.DPGaussianHistogram(domain_size=4)
    with self.assertRaises(ValueError):
      _ = mech.dp_event

  def test_call_raises_before_calibration(self):
    mech = primitives.DPGaussianHistogram(domain_size=4)
    with self.assertRaises(ValueError):
      mech(self.rng, np.array([0, 1]))

  def test_dp_event_type(self):
    mech = primitives.DPGaussianHistogram(domain_size=4).calibrate(zcdp_rho=0.5)
    event = mech.dp_event
    self.assertIsInstance(event, dp_accounting.GaussianDpEvent)
    self.assertAlmostEqual(event.noise_multiplier, 1.0)


if __name__ == "__main__":
  absltest.main()

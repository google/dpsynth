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
import dp_accounting
from dpsynth.local_mode import primitives
import numpy as np


class QuantilesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_median_basic(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    med = primitives._median(
        self.rng, data, lower=0.0, upper=10.0, zcdp_rho=100
    )
    self.assertBetween(med, 2.0, 4.5)

  def test_median_empty(self):
    data = np.array([])
    med = primitives._median(self.rng, data, lower=0.0, upper=10.0, epsilon=1.0)
    self.assertBetween(med, 0.0, 10.0)

  def test_quantiles_basic(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    eps_levels = primitives._quantile_epsilon_levels(100.0, 4)
    qs = primitives._quantiles(
        self.rng, data, lower=0.0, upper=10.0, epsilon_levels=eps_levels
    )
    self.assertLen(qs, 3)
    self.assertTrue(all(qs[i] <= qs[i + 1] for i in range(len(qs) - 1)))

  def test_quantiles_empty(self):
    data = np.array([])
    eps_levels = primitives._quantile_epsilon_levels(1.0, 4)
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
    eps_levels = primitives._quantile_epsilon_levels(100, 4)
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
    eps_levels = primitives._quantile_epsilon_levels(0.0, 4)
    qs = primitives._quantiles(
        self.rng, data, lower=0.0, upper=10.0, epsilon_levels=eps_levels
    )
    self.assertLen(qs, 3)
    self.assertTrue(all(0.0 <= q <= 10.0 for q in qs))

  def test_quantiles_zcdp_rho_inf(self):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    eps_levels = primitives._quantile_epsilon_levels(np.inf, 4)
    qs = primitives._quantiles(
        self.rng, data, lower=0.0, upper=10.0, epsilon_levels=eps_levels
    )
    self.assertLen(qs, 3)
    self.assertEqual(qs, [2.5, 4.0, 6.0])


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
    data = np.array([1] * 10 + [2])
    user_ids = np.array([1] * 10 + [2])
    selected, counts, sigma = primitives._select_partitions_sips(
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
    ) = sigma = primitives._select_partitions_sips(
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
      primitives._select_partitions_sips(
          self.rng, data, gdp_budget=10.0, delta=1e-5, user_ids=user_ids
      )


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


class DPQuantilesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  def test_calibrate_and_call(self):
    mech = primitives.DPQuantiles(lower=0.0, upper=10.0, num_partitions=4)
    calibrated = mech.calibrate(zcdp_rho=100.0)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = calibrated(self.rng, data)
    self.assertLen(result, 3)

  def test_direct_zcdp_rho(self):
    mech = primitives.DPQuantiles(
        lower=0.0, upper=10.0, num_partitions=4, zcdp_rho=100.0
    )
    result = mech(self.rng, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    self.assertLen(result, 3)

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
    self.assertLen(result, 4)
    np.testing.assert_allclose(result, [2, 3, 1, 0], atol=5.0)

  def test_direct_sigma(self):
    mech = primitives.DPGaussianHistogram(domain_size=3, sigma=0.0)
    data = np.array([0, 0, 1, 2, 2, 2])
    np.testing.assert_array_equal(mech(self.rng, data), [2, 1, 3])

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

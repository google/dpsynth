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
from dpsynth import TabularSynthesizer
from dpsynth import domain as dps_domain
from dpsynth.discrete_mechanisms import independent
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import aim_gdp
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np
import pandas as pd


def _make_discrete_data(rng, n=1000):
  domain = mbi.Domain(["a", "b", "c"], [3, 3, 3])
  a = rng.integers(0, 3, size=n)
  b = np.where(rng.random(n) < 0.75, a, rng.integers(0, 3, size=n))
  c = (a + b + rng.integers(0, 2, size=n)) % 3
  return mbi.Dataset({"a": a, "b": b, "c": c}, domain)


def _make_mixed_data(rng, n=1000):
  domains = {
      "a": dps_domain.CategoricalAttribute(possible_values=[0, 1, 2]),
      "b": dps_domain.CategoricalAttribute(possible_values=[0, 1, 2]),
      "c": dps_domain.NumericalAttribute(
          min_value=0.0, max_value=1.0, dtype="float"
      ),
  }
  a = rng.integers(0, 3, size=n)
  b = np.where(rng.random(n) < 0.75, a, rng.integers(0, 3, size=n))
  c = np.clip((a + b) / 4.0 + rng.normal(0.0, 0.1, size=n), 0.0, 1.0)
  return pd.DataFrame({"a": a, "b": b, "c": c}), domains


def _normalized_l1(data, model, clique):
  expected = data.project(clique).datavector().astype(float)
  actual = model.project(clique).datavector().astype(float)
  expected /= expected.sum()
  actual /= actual.sum()
  return np.abs(expected - actual).sum() / 2.0


def _discrete_workload_mechanism_baseline_errors(
    config, baseline_config, workload, zcdp_rho=5.0
):
  rng = np.random.default_rng(0)
  data = _make_discrete_data(rng)

  mechanism_result = config.configure(zcdp_rho=zcdp_rho)(rng, data)
  baseline_result = baseline_config.configure(zcdp_rho=zcdp_rho)(rng, data)

  mechanism_error = np.mean(
      [_normalized_l1(data, mechanism_result.model, clique) for clique in workload]
  )
  baseline_error = np.mean(
      [_normalized_l1(data, baseline_result.model, clique) for clique in workload]
  )
  return mechanism_error, baseline_error


def _mixed_workload_mechanism_baseline_errors(
    config, baseline_config, workload, zcdp_rho=5.0, numerical_bins=16
):
  rng = np.random.default_rng(0)
  data, domains = _make_mixed_data(rng, n=1000)

  mechanism_synth = TabularSynthesizer(
      domains=domains,
      discrete_mechanism=config,
      numerical_bins=numerical_bins,
  )
  baseline_synth = TabularSynthesizer(
      domains=domains,
      discrete_mechanism=baseline_config,
      numerical_bins=numerical_bins,
  )

  mechanism_result = mechanism_synth.configure(zcdp_rho=zcdp_rho)(rng, data)
  baseline_result = baseline_synth.configure(zcdp_rho=zcdp_rho)(rng, data)

  mechanism_error = np.mean([
      _normalized_l1(
          mechanism_result.codec.encode(data),
          mechanism_result.discrete_mechanism_result.model,
          clique,
      )
      for clique in workload
  ])
  baseline_error = np.mean([
      _normalized_l1(
          baseline_result.codec.encode(data),
          baseline_result.discrete_mechanism_result.model,
          clique,
      )
      for clique in workload
  ])
  return mechanism_error, baseline_error


class AIMTest(absltest.TestCase):

  def test_fits_one_way_marginals_with_aim(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)
    workload = [("a",), ("b",), ("c",)]
    config = aim.AIMMechanism(workload=workload, max_rounds=4, pgm_iters=500)

    calibrated = config.configure(zcdp_rho=10000)
    result = calibrated(np.random.default_rng(0), data)

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertNotEmpty(result.measurements)
    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_fits_one_way_marginals_with_aim_gdp(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)
    workload = [("a",), ("b",), ("c",)]

    config = aim_gdp.AIMGDPMechanism(
        workload=workload, max_rounds=4, pgm_iters=500
    )
    calibrated = config.configure(zcdp_rho=10000)
    result = calibrated(np.random.default_rng(0), data)

    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertNotEmpty(result.measurements)
    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = result.model.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_uncalibrated_aim_raises(self):
    config = aim.AIMMechanism()
    with self.assertRaisesRegex(ValueError, "calibrate"):
      _ = config.dp_event
    data = mbi.Dataset.synthetic(mbi.Domain(["a"], [3]), N=10)
    with self.assertRaisesRegex(ValueError, "calibrate"):
      config(np.random.default_rng(0), data)

  def test_uncalibrated_aim_gdp_raises(self):
    config = aim_gdp.AIMGDPMechanism()
    with self.assertRaisesRegex(ValueError, "calibrate"):
      _ = config.dp_event
    data = mbi.Dataset.synthetic(mbi.Domain(["a"], [3]), N=10)
    with self.assertRaisesRegex(ValueError, "calibrate"):
      config(np.random.default_rng(0), data)

  def test_discrete_workload_regression_with_aim(self):
    workload = [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")]
    config = aim.AIMMechanism(workload=workload, max_rounds=4, pgm_iters=500)
    baseline_config = independent.IndependentMechanism(pgm_iters=500)
    mechanism_error, baseline_error = (
      _discrete_workload_mechanism_baseline_errors(
        config, baseline_config, workload
      )
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_discrete_workload_regression_with_aim_gdp(self):
    workload = [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")]
    config = aim_gdp.AIMGDPMechanism(
      workload=workload, max_rounds=4, pgm_iters=500
    )
    baseline_config = independent.IndependentMechanism(pgm_iters=500)
    mechanism_error, baseline_error = (
      _discrete_workload_mechanism_baseline_errors(
        config, baseline_config, workload
      )
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_mixed_workload_regression_with_aim(self):
    workload = [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")]
    config = aim.AIMMechanism(workload=workload, max_rounds=4, pgm_iters=500)
    baseline_config = independent.IndependentMechanism(pgm_iters=500)
    mechanism_error, baseline_error = (
      _mixed_workload_mechanism_baseline_errors(
        config, baseline_config, workload
      )
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_mixed_workload_regression_with_aim_gdp(self):
    workload = [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")]
    config = aim_gdp.AIMGDPMechanism(
      workload=workload, max_rounds=4, pgm_iters=500
    )
    baseline_config = independent.IndependentMechanism(pgm_iters=500)
    mechanism_error, baseline_error = (
      _mixed_workload_mechanism_baseline_errors(
        config, baseline_config, workload
      )
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_default_configuration_values(self):
    config = aim.AIMMechanism()
    self.assertEqual(config.pgm_iters, 1000)

    gdp_config = aim_gdp.AIMGDPMechanism()
    self.assertEqual(gdp_config.pgm_iters, 1000)


if __name__ == "__main__":
  absltest.main()

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

"""Property tests shared across all discrete mechanisms."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import aim_gdp
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import direct
from dpsynth.discrete_mechanisms import independent
from dpsynth.discrete_mechanisms import mst
from dpsynth.discrete_mechanisms import swift
import mbi
import numpy as np

_ZCDP_RHO = 10000
_WORKLOAD = [('a', 'b'), ('b', 'c'), ('a',), ('b',), ('c',)]

_MECHANISMS = {
    'AIM': aim.AIMMechanism(workload=_WORKLOAD, max_rounds=4, pgm_iters=500),
    'AIM_GDP': aim_gdp.AIMGDPMechanism(
        workload=_WORKLOAD, max_rounds=4, pgm_iters=500
    ),
    'MST': mst.MSTMechanism(pgm_iters=500),
    'SWIFT': swift.SWIFTMechanism(workload=_WORKLOAD, pgm_iters=500),
    'Independent': independent.IndependentMechanism(pgm_iters=500),
    'Direct': direct.DirectMechanism(
        prespecified_marginal_queries=_WORKLOAD, pgm_iters=500
    ),
}


def _make_skewed_dataset(rng):
  """Creates a dataset where column 'a' concentrates in 3 of 10 bins."""
  domain = mbi.Domain(['a', 'b', 'c'], [10, 4, 5])
  df = {col: rng.integers(0, domain[col], size=1000) for col in domain}
  df['a'] = rng.choice(3, size=1000)  # Only bins 0-2 populated.
  return mbi.Dataset(df, domain)


class SupportingCliquesSufficiencyTest(parameterized.TestCase):
  """Checks that supporting_cliques are sufficient for each mechanism.

  For each mechanism, we:
    1. Compute supporting_cliques(domain).
    2. Build a CliqueVector from the true data projected onto those cliques.
    3. Run the mechanism using the CliqueVector as input data.
    4. Assert it completes without error — the CliqueVector supports every
       projection the mechanism needs.
  """

  @parameterized.named_parameters(*_MECHANISMS.items())
  def test_mechanism_runs_on_precomputed_marginals(self, mechanism):
    domain = mbi.Domain(['a', 'b', 'c', 'd'], [3, 4, 5, 6])
    data = mbi.Dataset.synthetic(domain, N=500)
    rng = np.random.default_rng(42)

    calibrated = mechanism.calibrate(zcdp_rho=_ZCDP_RHO)
    cliques = calibrated.supporting_cliques(domain)

    precomputed = mbi.CliqueVector.from_projectable(data, cliques)

    result = calibrated(rng, precomputed)
    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertIsNotNone(result.model)


class CompressionPropertyTest(parameterized.TestCase):
  """Tests that compression restores the original domain across mechanisms."""

  @parameterized.named_parameters(*_MECHANISMS.items())
  def test_compression_restores_domain(self, config):
    config = dataclasses.replace(config, compress_columns=True)
    rng = np.random.default_rng(0)
    data = _make_skewed_dataset(rng)
    original_domain = data.domain

    result = config.configure(zcdp_rho=_ZCDP_RHO)(rng, data)

    self.assertEqual(result.synthetic_data.domain, original_domain)

  @parameterized.named_parameters(*_MECHANISMS.items())
  def test_compression_with_initial_measurements(self, config):
    config = dataclasses.replace(config, compress_columns=True)
    rng = np.random.default_rng(0)
    data = _make_skewed_dataset(rng)
    original_domain = data.domain
    initial_measurements = common.measure_marginals_with_noise(
        rng, data, [('a',), ('b',)], gdp_sigma=1.0
    )

    mechanism = config.configure(zcdp_rho=_ZCDP_RHO)
    result = mechanism(rng, data, initial_measurements=initial_measurements)

    self.assertEqual(result.synthetic_data.domain, original_domain)
    self.assertNotEmpty(result.mappings)


class CalibrationTest(parameterized.TestCase):
  """Tests that calibration works across mechanisms."""

  @parameterized.named_parameters(*_MECHANISMS.items())
  def test_deprecated_zcdp_calibration(self, mechanism):
    rng = np.random.default_rng(0)
    data = _make_skewed_dataset(rng)
    result = mechanism.calibrate(zcdp_rho=_ZCDP_RHO)(rng, data)
    self.assertIsInstance(result, common.DiscreteMechanismResult)

  @parameterized.named_parameters(*_MECHANISMS.items())
  def test_zero_epsilon_calibration(self, mechanism):
    rng = np.random.default_rng(0)
    data = _make_skewed_dataset(rng)
    result = mechanism.calibrate(epsilon=0.0, delta=0.01)(rng, data)
    self.assertIsInstance(result, common.DiscreteMechanismResult)

  @parameterized.named_parameters(*_MECHANISMS.items())
  def test_low_epsilon_calibration(self, mechanism):
    self.skipTest('Low epsilon calibration is currently really slow, skipping.')
    rng = np.random.default_rng(0)
    data = _make_skewed_dataset(rng)
    result = mechanism.calibrate(epsilon=1e-3, delta=1e-5)(rng, data)
    self.assertIsInstance(result, common.DiscreteMechanismResult)


if __name__ == '__main__':
  absltest.main()

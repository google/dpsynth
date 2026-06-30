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

"""Property tests for supporting_cliques across all discrete mechanisms."""

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

_DOMAIN = mbi.Domain(['a', 'b', 'c', 'd'], [3, 4, 5, 6])
_WORKLOAD = [('a', 'b'), ('b', 'c'), ('c', 'd')]

_MECHANISMS = {
    'Independent': independent.IndependentMechanism(pgm_iters=100),
    'MST': mst.MSTMechanism(pgm_iters=100),
    'AIM': aim.AIMMechanism(workload=_WORKLOAD, max_rounds=2, pgm_iters=100),
    'AIMGDP': aim_gdp.AIMGDPMechanism(
        workload=_WORKLOAD, max_rounds=2, pgm_iters=100
    ),
    'SWIFT': swift.SWIFTMechanism(workload=_WORKLOAD, pgm_iters=100),
    'Direct': direct.DirectMechanism(
        prespecified_marginal_queries=_WORKLOAD, pgm_iters=100
    ),
}


class SupportingCliquesSufficiencyTest(parameterized.TestCase):
  """Checks that supporting_cliques are sufficient for each mechanism.

  For each mechanism, we:
    1. Compute supporting_cliques(domain).
    2. Build a CliqueVector from the true data projected onto those cliques.
    3. Run the mechanism using the CliqueVector as input data.
    4. Assert it completes without error — the CliqueVector supports every
       projection the mechanism needs.
  """

  @parameterized.named_parameters(
      *[(name, mech) for name, mech in _MECHANISMS.items()]
  )
  def test_mechanism_runs_on_precomputed_marginals(self, mechanism):
    data = mbi.Dataset.synthetic(_DOMAIN, N=500)
    rng = np.random.default_rng(42)

    calibrated = mechanism.calibrate(zcdp_rho=10_000)
    cliques = calibrated.supporting_cliques(_DOMAIN)

    # Build a CliqueVector that holds exactly the supporting cliques.
    precomputed = mbi.CliqueVector.from_projectable(data, cliques)

    # The mechanism should run to completion using only these marginals.
    result = calibrated(rng, precomputed)
    self.assertIsInstance(result, common.DiscreteMechanismResult)
    self.assertIsNotNone(result.model)


if __name__ == '__main__':
  absltest.main()

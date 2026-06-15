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
from dpsynth.discrete_mechanisms import independent
import mbi
import numpy as np


class IndependentTest(absltest.TestCase):

  def test_fits_one_way_marginals(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)

    config = independent.IndependentConfig(pgm_iters=500)
    synthetic = independent.run_mechanism(data, config, zcdp_rho=10000)

    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = synthetic.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=0.1)

  def test_duplicate_one_way_cliques_with_initial_potentials(self):
    # Regression test: when `initial_potentials` is provided (e.g. the empty
    # CliqueVector for the no-constraints case) and `initial_measurements`
    # already holds the one-way marginals, the cliques measured in the loop
    # duplicate them. `CliqueVector.expand` rejects duplicate cliques, so the
    # mechanism used to crash with "Cliques must be unique.".
    domain = mbi.Domain(["a", "b", "c"], [3, 4, 5])
    data = mbi.Dataset.synthetic(domain, N=1000)

    initial_measurements = [
        mbi.LinearMeasurement(data.project((col,)).datavector(), (col,))
        for col in data.domain
    ]
    initial_potentials = mbi.CliqueVector(domain, [], {})

    config = independent.IndependentConfig(pgm_iters=500)
    synthetic = independent.run_mechanism(
        data,
        config,
        zcdp_rho=10000,
        initial_measurements=initial_measurements,
        initial_potentials=initial_potentials,
    )

    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = synthetic.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=0.1)


if __name__ == "__main__":
  absltest.main()

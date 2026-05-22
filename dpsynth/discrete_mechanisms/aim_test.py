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
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import aim_gdp
import mbi
import numpy as np


class AIMTest(absltest.TestCase):

  def test_fits_one_way_marginals_with_aim(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)
    workload = [("a",), ("b",), ("c",)]
    config = aim.AIMConfig(workload=workload, max_rounds=4, pgm_iters=500)

    synthetic = aim.run_mechanism(data, config, zcdp_rho=10000)

    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = synthetic.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)

  def test_fits_one_way_marginals_with_aim_gdp(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)
    workload = [("a",), ("b",), ("c",)]

    config = aim_gdp.AIMGDPConfig(
        workload=workload, max_rounds=4, pgm_iters=500
    )
    synthetic = aim_gdp.run_mechanism(data, config, zcdp_rho=10000)

    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = synthetic.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)


if __name__ == "__main__":
  absltest.main()

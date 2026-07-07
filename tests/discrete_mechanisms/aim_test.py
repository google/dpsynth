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
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


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


if __name__ == "__main__":
  absltest.main()

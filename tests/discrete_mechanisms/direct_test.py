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
from dpsynth.discrete_mechanisms import direct
import mbi
import numpy as np


class DirectTest(absltest.TestCase):

  def test_fits_one_way_marginals(self):
    data = mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=1000)

    config = direct.DirectMechanism(
        prespecified_marginal_queries=[
            ('a', 'b'),
            ('a', 'c'),
            ('b', 'c'),
        ],
        pgm_iters=500,
    )
    synthetic = config.calibrate(zcdp_rho=10000)(np.random.default_rng(0), data)

    for col in data.domain:
      expected = data.project([col]).datavector()
      actual = synthetic.project([col]).datavector()
      np.testing.assert_allclose(actual, expected, atol=1)


if __name__ == '__main__':
  absltest.main()

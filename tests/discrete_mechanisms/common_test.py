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

import pickle

from absl.testing import absltest
from dpsynth import transformations
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


def assert_serializable(obj):
  pickle.dumps(obj)


class CommonTest(absltest.TestCase):

  def test_exponential_mechanism(self):
    rng = np.random.default_rng(0)
    scores = np.array([5, 20, -10, 3])
    idx = common.exponential_mechanism(
        scores, epsilon=1.0, sensitivity=1.0, rng=rng
    )
    self.assertIn(idx, [0, 1, 2, 3])
    idx = common.exponential_mechanism(
        scores, epsilon=1.0, sensitivity=1e-8, rng=rng
    )
    self.assertEqual(idx, 1)
    idx = common.exponential_mechanism(
        scores, epsilon=1e8, sensitivity=1.0, rng=rng
    )
    self.assertEqual(idx, 1)

  def test_measure_marginals_with_noise(self):
    data = mbi.Dataset.synthetic(mbi.Domain(["a", "b", "c"], [3, 4, 5]), N=1000)
    marginal_queries = [("a",), ("b",), ("c",)]
    measurements = common.measure_marginals_with_noise(
        np.random.default_rng(0), data, marginal_queries, gdp_sigma=1.0
    )
    self.assertLen(measurements, 3)
    for m in measurements:
      self.assertLen(m.clique, 1)

  def test_compressed_measurement(self):
    answer = np.array([5, 10, 15, 20, 25, 30])
    measurement = mbi.LinearMeasurement(answer, ("a",), 1.0)

    size, transform = transformations.create_rare_value_merging_transformation(
        np.array([True, False, True, False, True, False])
    )
    compressed = common.compressed_measurement(measurement, size, transform)
    self.assertEqual(compressed.clique, ("a",))
    self.assertLen(compressed.noisy_measurement, 4)
    assert_serializable(compressed)

  def test_get_domain_compression_transformations(self):
    answer = np.array([5, 10, 15, 20, 25, 30])
    measurement = mbi.LinearMeasurement(answer, ("a",), stddev=4.0)
    compressed_domain = common.get_domain_compression_transformations(
        [measurement]
    )[0]
    self.assertEqual(compressed_domain, mbi.Domain.fromdict({"a": 5}))


if __name__ == "__main__":
  absltest.main()

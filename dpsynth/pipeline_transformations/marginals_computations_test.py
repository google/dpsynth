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
import apache_beam as beam
from apache_beam.testing import test_pipeline
from dpsynth.pipeline_transformations import marginals_computations
import mbi
import numpy as np
import pipeline_dp


class MarginalsTest(absltest.TestCase):

  def test_beam_backend(self):
    with test_pipeline.TestPipeline() as pipeline:
      data = [(0, 1, 2), (0, 0, 2), (0, 2, 2), (1, 1, 2)]

      queries = pipeline | "Create queries" >> beam.Create([[(0, 1), (2,)]])
      domain = pipeline | "Create domain" >> beam.Create(
          [mbi.Domain([0, 1, 2], [2, 3, 4])]
      )
      data = pipeline | "Create" >> beam.Create(data)
      backend = pipeline_dp.BeamBackend()
      marginals = marginals_computations.compute_exact_marginals(
          backend, data, queries, domain
      )

    self.assertIsInstance(marginals, beam.PCollection)

  def test_compute_exact_marginals(self):
    backend = pipeline_dp.LocalBackend()

    data = [(0, 1, 2), (0, 0, 2), (0, 2, 2), (1, 1, 2)]

    queries = [[(0, 1), (2,)]]  # singleton list
    domain = [mbi.Domain([0, 1, 2], [2, 3, 4])]
    marginals = dict(
        list(
            marginals_computations.compute_exact_marginals(
                backend, data, queries, domain
            )
        )
    )
    self.assertEqual(marginals.keys(), {(0, 1), (2,)})

    np.testing.assert_equal(
        marginals[(0, 1)], np.array([[1, 1, 1], [0, 1, 0]]).ravel()
    )
    np.testing.assert_equal(marginals[(2,)], np.array([0, 0, 4, 0]))

  def test_add_dp_noise_to_marginals(self):
    # Create test data
    marginals_data = [
        ((0,), np.array([10.0, 20.0])),
        ((1,), np.array([30.0, 40.0])),
    ]
    backend = pipeline_dp.LocalBackend()
    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=100.0, total_delta=1e-5
    )
    dp_engine = pipeline_dp.DPEngine(
        budget_accountant=accountant,
        backend=backend,
    )
    max_partitions_contributed = 2

    # Add DP noise to marginals
    linear_measurements = marginals_computations.add_dp_noise_to_marginals(
        backend, dp_engine, marginals_data, max_partitions_contributed
    )
    accountant.compute_budgets()
    linear_measurements = list(linear_measurements)

    # Assert the result
    self.assertLen(linear_measurements, 2)
    self.assertEqual(linear_measurements[0].clique, (0,))
    self.assertNotEqual(linear_measurements[0].noisy_measurement[0], 10.0)
    np.testing.assert_allclose(
        linear_measurements[0].noisy_measurement, [10.0, 20.0], atol=1.0
    )
    self.assertEqual(linear_measurements[1].clique, (1,))
    self.assertNotEqual(linear_measurements[1].noisy_measurement[1], 40.0)
    np.testing.assert_allclose(
        linear_measurements[1].noisy_measurement, [30.0, 40.0], atol=1.0
    )

  def create_dp_engine(self):
    accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=200, total_delta=0.1
    )
    backend = pipeline_dp.LocalBackend()
    return pipeline_dp.DPEngine(accountant, backend), accountant

  def test_one_way_dp_marginals_empty_input(self):
    backend = pipeline_dp.LocalBackend()
    data = []
    domain = [mbi.Domain(["a", "b", "c", "d"], [3, 4, 5, 6])]
    dp_engine, accountant = self.create_dp_engine()
    result = marginals_computations.compute_one_way_dp_marginals(
        backend, dp_engine, data, domain, 1
    )
    accountant.compute_budgets()
    result = list(result)[0]

    self.assertLen(result, 4)
    np.testing.assert_allclose(result[0].noisy_measurement, [0, 0, 0], atol=0.5)

  def test_one_way_dp_marginals_single_row(self):
    backend = pipeline_dp.LocalBackend()
    dp_engine, accountant = self.create_dp_engine()
    data = [(0, 1)]
    domain = [mbi.Domain(["col0", "col1"], [2, 2])]
    result = marginals_computations.compute_one_way_dp_marginals(
        backend, dp_engine, data, domain, 2
    )
    accountant.compute_budgets()
    result = list(result)[0]

    self.assertLen(result, 2)
    np.testing.assert_allclose(result[0].noisy_measurement, [1, 0], atol=0.5)
    np.testing.assert_allclose(result[1].noisy_measurement, [0, 1], atol=0.5)

  def test_one_way_dp_marginals_multiple_rows_different_values(self):
    backend = pipeline_dp.LocalBackend()
    dp_engine, accountant = self.create_dp_engine()
    data = [(0, 1, 2), (1, 0, 3)]
    domain = [mbi.Domain(["col0", "col1", "col2"], [3, 4, 5])]
    result = marginals_computations.compute_one_way_dp_marginals(
        backend, dp_engine, data, domain, 3
    )
    accountant.compute_budgets()
    result = list(result)[0]

    self.assertLen(result, 3)
    np.testing.assert_allclose(result[0].noisy_measurement, [1, 1, 0], atol=0.5)
    np.testing.assert_allclose(
        result[1].noisy_measurement, [1, 1, 0, 0], atol=0.5
    )
    np.testing.assert_allclose(
        result[2].noisy_measurement, [0, 0, 1, 1, 0], atol=0.5
    )

  def test_one_way_dp_marginals_multiple_rows_repeated_values(self):
    backend = pipeline_dp.LocalBackend()
    dp_engine, accountant = self.create_dp_engine()
    data = [(0, 2), (0, 2), (2, 0), (0, 2)]
    domain = [mbi.Domain(["col0", "col1"], [3, 3])]
    result = marginals_computations.compute_one_way_dp_marginals(
        backend, dp_engine, data, domain, 2
    )
    accountant.compute_budgets()
    result = list(result)[0]

    self.assertLen(result, 2)
    np.testing.assert_allclose(result[0].noisy_measurement, [3, 0, 1], atol=0.5)
    np.testing.assert_allclose(result[1].noisy_measurement, [1, 0, 3], atol=0.5)

  def test_combine_marginals(self):
    backend = pipeline_dp.LocalBackend()
    one_way_marginals = [[
        mbi.LinearMeasurement(
            noisy_measurement=np.array([1, 0, 0]),
            clique=(0,),
            stddev=1.0,
        ),
        mbi.LinearMeasurement(
            noisy_measurement=np.array([0, 1, 0]),
            clique=(1,),
            stddev=1.0,
        ),
        mbi.LinearMeasurement(
            noisy_measurement=np.array([0, 0, 1]),
            clique=(2,),
            stddev=1.0,
        ),
    ]]
    two_way_marginals = [
        mbi.LinearMeasurement(
            noisy_measurement=np.array([0, 2, 1, 4]),
            clique=(
                0,
                1,
            ),
            stddev=2.0,
        ),
        mbi.LinearMeasurement(
            noisy_measurement=np.array([0, 0, 1, 1]),
            clique=(
                0,
                2,
            ),
            stddev=3.0,
        ),
    ]
    result = marginals_computations.combine_marginals(
        backend, one_way_marginals, two_way_marginals
    )
    result = list(result)[0]
    self.assertLen(result, 5)
    np.testing.assert_equal(result[0].noisy_measurement, np.array([1, 0, 0]))
    np.testing.assert_equal(result[0].stddev, 1.0)
    np.testing.assert_equal(result[1].noisy_measurement, np.array([0, 1, 0]))
    np.testing.assert_equal(result[2].noisy_measurement, np.array([0, 0, 1]))
    np.testing.assert_equal(result[3].noisy_measurement, np.array([0, 2, 1, 4]))
    np.testing.assert_equal(result[4].noisy_measurement, np.array([0, 0, 1, 1]))
    np.testing.assert_equal(result[4].stddev, 3.0)
    np.testing.assert_equal(result[0].clique, (0,))
    np.testing.assert_equal(result[1].clique, (1,))
    np.testing.assert_equal(result[2].clique, (2,))
    np.testing.assert_equal(result[3].clique, (0, 1))
    np.testing.assert_equal(result[4].clique, (0, 2))


if __name__ == "__main__":
  absltest.main()

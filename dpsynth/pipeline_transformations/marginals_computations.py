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

"""Primitive DP building blocks."""

from collections.abc import Iterable

from dpsynth.pipeline_transformations import types
import mbi
import numpy as np
import pipeline_dp


def compute_exact_marginals(
    backend: pipeline_dp.PipelineBackend,
    data: types.Collection[types.Record],
    marginal_queries: types.Collection[types.Clique],
    domain: types.Collection[mbi.Domain],
) -> types.Collection[types.Marginal]:
  """Compute exact marginals over a collection of records.

  Example Usage:
  >>> backend = pipeline_dp.LocalBackend()
  >>> data = [(0, 1), (0, 0), (0, 0), (1, 1)]
  >>> queries = [(0,)]
  >>> domain = mbi.Domain([0, 1], [2, 3])
  >>> marginals = compute_exact_marginals(backend, data, queries, domain)
  >>> print(next(marginals))
  np.array([3, 1])

  Args:
    backend: The backend to use for running the pipeline operations.
    data: The collection of records to compute the marginals over.  Each record
      is a tuple of non-negative integers [0,1, ..., domain[i]-1].
    marginal_queries: The marginal queries to compute. Each clique is a tuple of
      attributes in the domain.
    domain: The domain of the marginals. Each attribute in the domain should be
      a non-negative integer that indexes into the record tuple.

  Returns:
    The marginals computed over the data.
  """

  def extract_clique_values(row, marginal_queries):
    return [(q, tuple(row[i] for i in q)) for q in marginal_queries]

  # feature_values: (col1, col2), (val1, val2)
  feature_values = backend.flat_map_with_side_inputs(
      data, extract_clique_values, [marginal_queries], 'Extract Clique Values'
  )

  # feature_counts: ((col1, col2), (val1, val2)), count
  feature_counts = backend.count_per_element(feature_values, 'Count Features')

  def reformat(cols_vals, count):
    cols, vals = cols_vals
    return cols, (vals, count)

  # feature_counts: (col1, col2), ((val1, val2)), count)
  feature_counts = backend.map_tuple(feature_counts, reformat, 'Reformat')

  # marginals: (col1, col2), list[((val1, val2), count)]
  marginals = backend.group_by_key(feature_counts, 'Group by Clique')

  def to_np_array(key_value_counts, domain):
    key, value_counts = key_value_counts
    result = np.zeros(domain.project(key).shape)
    for val, count in value_counts:
      result[val] = count

    return key, result.ravel()

  # marginals: (col1, col2), np.ndarray
  marginals = backend.map_with_side_inputs(
      marginals, to_np_array, [domain], 'Convert to Numpy'
  )
  return marginals


def compute_one_way_dp_marginals(
    backend: pipeline_dp.PipelineBackend,
    dp_engine: pipeline_dp.DPEngine,
    data: types.Collection[types.Record],
    domain: types.Collection[mbi.Domain],
    num_attributes: int,
) -> types.Collection[list[mbi.LinearMeasurement]]:
  """Computes one-way DP marginals.

  The output marginals are sorted by attribute index.

  Args:
    backend: The backend to use for running the pipeline operations.
    dp_engine: The DP engine.
    data: The input data, where each tuple represents a row and each element in
      the tuple represents a value in a column.
    domain: The mbi domain, defining the possible values for each column.
    num_attributes: The number of attributes in the data.

  Returns:
    A list of value counts for each column.
  """
  flat_data = backend.flat_map(
      data,
      # Applying directly enumerate failed when running on Beam.
      lambda x: enumerate(x),  # pylint: disable=unnecessary-lambda
      stage_name='Flat map to (index, value)',
  )

  params = pipeline_dp.AggregateParams(
      metrics=[pipeline_dp.Metrics.COUNT],
      noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
      max_partitions_contributed=num_attributes,
      max_contributions_per_partition=1,
      contribution_bounds_already_enforced=True,
      public_partitions_already_filtered=True,
      output_noise_stddev=True,
  )

  def create_public_partitions(domain: mbi.Domain) -> Iterable[tuple[int, int]]:
    for attribute_index, size in enumerate(domain.shape):
      for val in range(size):
        yield (attribute_index, val)

  data_extractor = pipeline_dp.DataExtractors(
      partition_extractor=lambda x: x,
      # 'value_extractor' is not needed for Count aggregation, but PipelineDP
      # requires value_extractor to be set. Make it always return 0.
      value_extractor=lambda _: 0,
  )

  public_partitions = backend.flat_map(
      domain,
      create_public_partitions,
      stage_name='Create public partitions',
  )
  dp_feature_counts = dp_engine.aggregate(
      flat_data, params, data_extractor, public_partitions
  )

  reformatted_feature_counts = backend.map_tuple(
      dp_feature_counts,
      lambda index_value, aggregate: (
          index_value[0],
          (index_value[1], aggregate.count, aggregate.count_noise_stddev),
      ),
      stage_name='Reformat to column: value-count pairs',
  )
  marginals = backend.group_by_key(
      reformatted_feature_counts, stage_name='Group by column'
  )

  def create_linear_measurements(
      column_data: tuple[int, Iterable[tuple[int, float, float]]],
      domain: mbi.Domain,
  ) -> mbi.LinearMeasurement:
    column_index, value_count_pairs = column_data
    result = np.zeros(domain.shape[column_index], dtype=np.float64)
    count_noise_stddev = 0.0
    for value, count, stddev in value_count_pairs:
      result[value] = count
      count_noise_stddev = stddev
    return mbi.LinearMeasurement(result, (column_index,), count_noise_stddev)  # pyrefly: ignore[bad-argument-type]

  linear_measurements = backend.map_with_side_inputs(
      marginals,
      create_linear_measurements,
      [domain],
      stage_name='Create LinearMeasurements',
  )

  linear_measurements = backend.to_list(
      linear_measurements, stage_name='To singleton list'
  )

  return backend.map(
      linear_measurements,
      lambda x: sorted(x, key=lambda x: x.clique),
      'Sort 1 way marginals',
  )


def add_dp_noise_to_marginals(
    backend: pipeline_dp.PipelineBackend,
    dp_engine: pipeline_dp.DPEngine,
    marginals: types.Collection[types.Marginal],
    number_of_marginals: int,
) -> types.Collection[mbi.LinearMeasurement]:
  """Adds DP noise to the marginals.

  Args:
    backend:  The backend to use for running the pipeline operations.
    dp_engine: The DP engine to perform the noise addition.
    marginals: The marginals to add DP noise to.
    number_of_marginals: The number of marginals one record can contribute to.

  Returns:
    The marginals with DP noise added.
  """
  params = pipeline_dp.AddDPNoiseParams(
      l2_sensitivity=np.sqrt(number_of_marginals),
      noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
      output_noise_stddev=True,
  )
  noised_marginals = dp_engine.add_dp_noise(marginals, params)
  # noised_marginals: (clique, np.ndarray)

  linear_measurements = backend.map_tuple(
      noised_marginals,
      lambda clique, marginal_stddev: mbi.LinearMeasurement(
          noisy_measurement=marginal_stddev[0],
          clique=clique,
          stddev=marginal_stddev[1],
      ),
      'Convert to Linear Measurements',
  )
  return linear_measurements


def combine_marginals(
    backend: pipeline_dp.PipelineBackend,
    one_way_marginals: types.Collection[list[mbi.LinearMeasurement]],
    two_way_marginals: types.Collection[mbi.LinearMeasurement],
) -> types.Collection[list[mbi.LinearMeasurement]]:
  """Combines one way and two way marginals.

  Args:
    backend: The backend to use for running the pipeline operations.
    one_way_marginals: The one way marginals. It is a singleton collection of
      list of one way marginals. Is is aligned with the output of
      compute_one_way_dp_marginals.
    two_way_marginals: The two way marginals. The format is aligned with the
      output of add_dp_noise_to_marginals.

  Returns:
    The singleton collection of combined marginals.
  """
  one_way_marginals_unnested = backend.flat_map(
      one_way_marginals, lambda x: x, 'Unnest one way marginals'
  )  # (mbi.LinearMeasurement,)

  marginals = backend.flatten(
      [
          one_way_marginals_unnested,
          two_way_marginals,
      ],
      'Combine 1d and 2d marginals',
  )  # (mbi.LinearMeasurement,)

  return backend.to_list(marginals, 'ToList')


def compute_errors(
    backend: pipeline_dp.PipelineBackend,
    one_way_dp_marginals: types.Collection[list[mbi.LinearMeasurement]],
    exact_marginals: types.Collection[tuple[types.Clique, np.ndarray]],
) -> types.Collection[dict[types.Clique, float]]:
  """Computes errors for the SWIFT/MST mechanism.

  Errors are L1 vector norm of the difference between the marginal of the
  independent attributes and the exact marginal of the clique.

  Args:
    backend: The backend to use for running the pipeline operations.
    one_way_dp_marginals: One-way marginals computed with DP for each attribute.
    exact_marginals: Exact marginals for each clique.

  Returns:
    A singleton collection of a dictionary mapping each clique to
    its corresponding error.
  """

  def compute_errors_fn(
      exact_marginal: tuple[types.Clique, np.ndarray],
      one_way_dp_marginals: list[mbi.LinearMeasurement],
  ) -> tuple[types.Clique, float]:
    clique, exact_vals = exact_marginal
    estimated_dataset_size = np.mean(
        [np.sum(m.noisy_measurement) for m in one_way_dp_marginals]
    )

    # Get 1D marginals for attributes in the clique
    marginals = [one_way_dp_marginals[a].noisy_measurement for a in clique]

    # Compute outer product of all marginals
    letters = 'abcdefghijklmnopqrstuvwxyz'[: len(clique)]
    formula = ','.join(letters) + '->' + ''.join(letters)
    marginal_of_independent = np.einsum(formula, *marginals)
    # Scale back
    marginal_of_independent = marginal_of_independent / (
        estimated_dataset_size ** (len(clique) - 1)
    )

    l1_vector_norm = np.sum(
        np.abs(marginal_of_independent.ravel() - exact_vals)
    )
    return clique, l1_vector_norm

  errors = backend.map_with_side_inputs(
      exact_marginals,
      compute_errors_fn,
      [one_way_dp_marginals],
      'Compute Errors',
  )  # (Clique, float)

  errors_singleton = backend.to_list(errors, 'To List')

  return backend.map(errors_singleton, dict, 'Pack to Dict')

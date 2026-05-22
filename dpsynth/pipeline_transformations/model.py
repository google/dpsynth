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

"""Model fitting and inference for synthetic tabular data generation in Beam pipelines."""  # pylint: disable=line-too-long

from dpsynth.pipeline_transformations import types
import mbi
import pipeline_dp


def fit_model(
    backend: pipeline_dp.PipelineBackend,
    linear_measurements: types.Collection[list[mbi.LinearMeasurement]],
    domain: types.Collection[mbi.Domain],
) -> types.Collection[mbi.MarkovRandomField]:
  """Fits a model to the given linear measurements.

  Args:
    backend: The backend to use for data generation.
    linear_measurements: The singleton collection of the private marginal
      measurements of the dataset.
    domain: The singleton collection of the domain of the marginals.

  Returns:
    The singleton collection with the fitted model.
  """

  def fit_model_fn(
      linear_measurements: list[mbi.LinearMeasurement],
      domain: mbi.Domain,
  ):
    return mbi.estimation.mirror_descent(
        domain, linear_measurements, iters=2500
    )

  return backend.map_with_side_inputs(
      linear_measurements, fit_model_fn, [domain], 'Fit Model'
  )


def generate_synthetic_data(
    backend: pipeline_dp.PipelineBackend,
    model: types.Collection[mbi.MarkovRandomField],
    domain: types.Collection[mbi.Domain],
    num_records: int | None = None,
    max_records_per_task: int = 10**6,
) -> types.Collection[tuple[int, ...]]:
  """Generates synthetic data from the model.

  Args:
    backend: The backend to use for running the pipeline operations.
    model: The model to generate data from.
    domain: The domain to generate data from.
    num_records: The number of records to generate. If None, the number of
      generated records is equal to 'model.total'.
    max_records_per_task: The maximum number of records to generate per task.

  Returns:
    types.Collection with tuples, where each tuple is a record of the synthetic
    discretized dataset. The ordering of elements in the tuple corresponds
    to the ordering of attributes in the domain.
  """
  # Scalable generation performed by splitting  num_records into smaller chunks
  # (tasks) and then applying map operation, with a mapper function which maps a
  # number 'n' to a tuple of 'n' synthetic records.
  # Note: Running the randomized rounding procedure k times for N/k records is
  # not the same as running it once for N records. In the future a different
  # implementation might be needed.
  if num_records is None:
    num_records_col = backend.map(model, lambda m: m.total, 'model.total')
  else:
    num_records_col = backend.to_collection([num_records], model, 'num_records')
  # num_records_per_task: (int)
  num_records_per_task = backend.flat_map(
      num_records_col,
      lambda n: _get_num_records_per_task(n, max_records_per_task),
      'num_records_per_task',
  )

  # Reshuffle the collection over workers to allow for more parallelism.
  num_records_per_task = backend.reshuffle(
      num_records_per_task, 'reshuffle num_records_per_task'
  )

  def gen_synthetic_data_task(num_records_per_task, model, domain):
    df = model.synthetic_data(rows=num_records_per_task).df

    # Convert the DataFrame to a list of tuples. The ordering of elements in
    # tuples correspond to the ordering of attributes in the domain.
    return df[list(domain.attributes)].itertuples(index=False, name=None)

  return backend.flat_map_with_side_inputs(
      num_records_per_task,
      gen_synthetic_data_task,
      [model, domain],
      'Generate synthetic data',
  )


def _get_num_records_per_task(
    num_records: int, max_records_per_task: int
) -> list[int]:
  """Splits the number of records to generate into multiple tasks."""
  # Maximum number of records to generate per task is chosen as 10**6 pretty
  # arbitrarily.
  # different value.
  result = []
  while num_records > 0:
    result.append(min(max_records_per_task, num_records))
    num_records -= result[-1]
  return result

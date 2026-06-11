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

"""The library for evaluating synthetic tabular data against original data."""

from typing import Any

import apache_beam as beam
from eval import attribute_statistics
from eval import correlation_computation
from eval import marginal_total_variance
from eval import one_way_distribution_computation
from eval import types
from dpsynth.pipeline_transformations import diagnostic_info
import pipeline_dp


def _get_backend(
    collection: types.Collection[types.Record],
) -> pipeline_dp.PipelineBackend:
  """Returns the appropriate PipelineBackend based on the collection type."""

  if isinstance(collection, beam.PCollection):
    return pipeline_dp.BeamBackend()
  return pipeline_dp.LocalBackend()


def _add_key(
    backend: pipeline_dp.PipelineBackend, col: types.Collection[Any], key: int
) -> types.Collection[tuple[int, Any]]:
  """Adds a key to each element in the collection."""
  return backend.map(col, lambda record: (key, record), "AddKey")


def _create_eval_report(
    backend: pipeline_dp.PipelineBackend,
    original_dataset_statistics: types.Collection[
        diagnostic_info.DatasetStatistics
    ],
    synthetic_dataset_statistics: types.Collection[
        diagnostic_info.DatasetStatistics
    ],
) -> diagnostic_info.TabularEvalReport:
  """Creates a eval report from report components."""
  original_dataset_statistics = _add_key(
      backend, original_dataset_statistics, 0
  )
  synthetic_dataset_statistics = _add_key(
      backend, synthetic_dataset_statistics, 1
  )

  col = backend.flatten(
      [original_dataset_statistics, synthetic_dataset_statistics],
      "FlattenDatasetStatistics",
  )
  # (0, original_dataset_statistics)
  # (1, synthetic_dataset_statistics)
  col = backend.to_list(col, "ToList")

  # [(0, original_dataset_statistics), (1, synthetic_dataset_statistics)]
  def create_eval_report(
      components: list[tuple[int, diagnostic_info.DatasetStatistics]],
  ) -> diagnostic_info.TabularEvalReport:
    original_dataset_statistics = synthetic_dataset_statistics = None
    for i, component in components:
      if i == 0:
        original_dataset_statistics = component
      elif i == 1:
        synthetic_dataset_statistics = component
      else:
        raise ValueError(f"Unexpected id of component: {i}")
    return diagnostic_info.TabularEvalReport(
        original_dataset_statistics=original_dataset_statistics,
        synthetic_dataset_statistics=synthetic_dataset_statistics,
    )

  return backend.map(col, create_eval_report, "CreateEvalReport")


def evaluate(
    original_data: types.Collection[types.Record],
    synthetic_data: types.Collection[types.Record],
    config: diagnostic_info.TabularEvalConfig,
    backend: pipeline_dp.PipelineBackend | None = None,
) -> types.Collection[diagnostic_info.TabularEvalReport]:
  """Evaluates synthetic tabular data against original data.

  Args:
    original_data: Collection of original records.
    synthetic_data: Collection of synthetic records.
    config: Evaluation configuration.
    backend: Optional PipelineBackend to use. If not provided, it will be
      inferred from the data.

  Returns:
    A collection containing a single TabularEvalReport.
  """
  if backend is None:
    backend = _get_backend(original_data)

  original_dataset_statistics, org_size = (
      attribute_statistics.compute_dataset_statistics(
          original_data, config, backend
      )
  )
  synthetic_dataset_statistics, syn_size = (
      attribute_statistics.compute_dataset_statistics(
          synthetic_data, config, backend
      )
  )

  eval_report = _create_eval_report(
      backend, original_dataset_statistics, synthetic_dataset_statistics
  )
  # Singleton: eval_report

  eval_report = (
      one_way_distribution_computation.compute_one_way_marginal_distance(
          eval_report, backend
      )
  )
  eval_report = correlation_computation.compute_correlation_distance(
      eval_report, backend
  )

  categorical_indices = [
      i
      for i, dtype in enumerate(config.attribute_types)
      if types.DataType(dtype).is_categorical()
  ]
  marginal_eval_reports = (
      marginal_total_variance.compute_multi_way_marginal_distance(
          original_data,
          synthetic_data,
          categorical_indices,
          org_size,
          syn_size,
          backend,
          max_marginal_degree=config.max_marginal_degree
          if config.max_marginal_degree
          else 2,
      )
  )
  marginal_eval_reports = backend.to_list(
      marginal_eval_reports, "MarginalReportsToList"
  )

  def add_marginals(
      report: diagnostic_info.TabularEvalReport,
      marginals_list: list[diagnostic_info.CategoricalMarginal],
  ) -> diagnostic_info.TabularEvalReport:
    new_report = diagnostic_info.TabularEvalReport()
    new_report.CopyFrom(report)
    new_report.marginal_eval_reports.extend(marginals_list)
    return new_report

  eval_report = backend.map_with_side_inputs(
      eval_report, add_marginals, [marginal_eval_reports], "AddMarginals"
  )

  return eval_report

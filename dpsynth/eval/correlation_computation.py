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

"""Computes two-way attribute correlations."""

from collections.abc import Iterable
import dataclasses
import itertools
from typing import Any
from dpsynth.eval import types
from dpsynth.pipeline_transformations import diagnostic_info
import numpy as np
import pipeline_dp
from scipy import stats


def compute_correlations(
    data: types.Collection[types.Record],
    config: diagnostic_info.TabularEvalConfig,
    backend: pipeline_dp.PipelineBackend,
) -> types.Collection[list[diagnostic_info.CorrelationValue]]:
  """Computes Cramer's V correlation for all pairs of categorical attributes.

  Args:
    data: Collection of records.
    config: Tabular evaluation configuration.
    backend: PipelineBackend to use for computations.

  Returns:
    A collection containing a list of CorrelationValue protos.
  """
  categorical_indices = [
      i
      for i, dtype in enumerate(config.attribute_types)
      if types.DataType(dtype).is_categorical()
  ]

  if len(categorical_indices) < 2:
    return backend.map(
        backend.to_list(backend.flatten([], "EmptyFlatten"), "ToList"),
        lambda _: [],
        "EmptyCorrelation",
    )

  def extract_pairs(
      record: types.Record,
  ) -> Iterable[tuple[tuple[int, int], tuple[Any, Any]]]:
    for i, j in itertools.combinations(categorical_indices, 2):
      yield ((i, j), (record[i], record[j]))

  pairs = backend.flat_map(data, extract_pairs, "ExtractCategoricalPairs")
  # ((i, j), (val_i, val_j))

  joint_counts = backend.count_per_element(pairs, "CountJointFrequencies")
  # (((i, j), (val_i, val_j)), count)

  def rekey_by_pair(item: tuple[tuple[tuple[int, int], tuple[Any, Any]], int]):
    (pair_indices, values), count = item
    return (pair_indices, (values[0], values[1], count))

  joint_counts_by_pair = backend.map(joint_counts, rekey_by_pair, "RekeyByPair")
  # ((i, j), (val_i, val_j, count))

  grouped_joint_counts = backend.group_by_key(
      joint_counts_by_pair, "GroupJointCountsByPair"
  )
  # ((i, j), [(val_i, val_j, count)])

  def compute_cramers_v(
      pair_indices: tuple[int, int], joint_counts: list[tuple[Any, Any, int]]
  ) -> diagnostic_info.CorrelationValue:
    i, j = pair_indices

    # Extract unique values for each attribute to build the contingency table
    vals_i = sorted(list(set(c[0] for c in joint_counts)), key=str)
    vals_j = sorted(list(set(c[1] for c in joint_counts)), key=str)

    val_to_idx_i = {val: idx for idx, val in enumerate(vals_i)}
    val_to_idx_j = {val: idx for idx, val in enumerate(vals_j)}

    table = np.zeros((len(vals_i), len(vals_j)))
    n = 0
    for val_i, val_j, count in joint_counts:
      table[val_to_idx_i[val_i], val_to_idx_j[val_j]] = count
      n += count

    if n == 0 or table.shape[0] <= 1 or table.shape[1] <= 1:
      v = 0.0
    else:
      chi2 = stats.chi2_contingency(table, correction=False)[0]
      v = np.sqrt(chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1)))

    return diagnostic_info.CorrelationValue(
        attribute_x=config.attributes[i],
        attribute_y=config.attributes[j],
        value=v,
    )

  correlations = backend.map_tuple(
      grouped_joint_counts, compute_cramers_v, "ComputeCramersV"
  )
  # CorrelationValue

  return backend.to_list(correlations, "CorrelationsToList")


def compute_correlation_distance(
    eval_report: types.Collection[diagnostic_info.TabularEvalReport],
    backend: pipeline_dp.PipelineBackend,
) -> types.Collection[diagnostic_info.TabularEvalReport]:
  """Computes L2 distance between original and synthetic correlation matrices.

  Args:
    eval_report: Collection containing the TabularEvalReport to update.
    backend: PipelineBackend to use for computations.

  Returns:
    A collection with the updated TabularEvalReport including correlation
    distance.
  """

  def compute_distance(
      report: diagnostic_info.TabularEvalReport,
  ) -> diagnostic_info.TabularEvalReport:
    orig_corrs = {
        (c.attribute_x, c.attribute_y): c.value
        for c in report.original_dataset_statistics.correlations
    }
    syn_corrs = {
        (c.attribute_x, c.attribute_y): c.value
        for c in report.synthetic_dataset_statistics.correlations
    }

    all_pairs = set(orig_corrs.keys()) | set(syn_corrs.keys())

    if not all_pairs:
      distance = 0.0
    else:
      squared_diff_sum = sum(
          (orig_corrs.get(pair, 0.0) - syn_corrs.get(pair, 0.0)) ** 2
          for pair in all_pairs
      )
      distance = np.sqrt(squared_diff_sum / len(all_pairs))

    new_report = dataclasses.replace(report, correlation_distance=distance)
    return new_report

  return backend.map(
      eval_report, compute_distance, "ComputeCorrelationDistance"
  )

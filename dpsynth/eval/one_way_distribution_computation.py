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

"""Computes one-way marginal distances."""

from collections.abc import Iterable
from typing import Any
from dpsynth.eval import types
from dpsynth.pipeline_transformations import diagnostic_info
import numpy as np
import pipeline_dp
from scipy import stats


# marginal_total_variance.py.
def compute_one_way_marginal_distance(
    eval_report: types.Collection[diagnostic_info.TabularEvalReport],
    backend: pipeline_dp.PipelineBackend,
) -> types.Collection[diagnostic_info.TabularEvalReport]:
  """Computes TV distance and Chi-squared p-values for categorical attributes.

  Args:
    eval_report: Collection containing the TabularEvalReport to update.
    backend: PipelineBackend to use for computations.

  Returns:
    A collection with the updated TabularEvalReport including attribute-level
    metrics.
  """

  def extract_attribute_statistics(
      eval_report: diagnostic_info.TabularEvalReport,
  ) -> Iterable[
      tuple[int, tuple[list[tuple[Any, int]], list[tuple[Any, int]]]]
  ]:
    original_statistics = (
        eval_report.original_dataset_statistics.attribute_statistics
    )
    synthetic_statistics = (
        eval_report.synthetic_dataset_statistics.attribute_statistics
    )
    n_attributes = len(original_statistics)
    assert len(synthetic_statistics) == n_attributes

    for i, (org, syn) in enumerate(
        zip(original_statistics, synthetic_statistics)
    ):
      if not org.HasField("categorical_statistics"):
        continue
      if not syn.HasField("categorical_statistics"):
        continue
      if org.attribute_name != syn.attribute_name:
        raise ValueError(
            "Attribute not found in original dataset: %s" % syn.attribute_name
        )
      org_counts = [
          (c.category, c.count)
          for c in org.categorical_statistics.category_counts
      ]
      syn_counts = [
          (c.category, c.count)
          for c in syn.categorical_statistics.category_counts
      ]
      yield (i, (org_counts, syn_counts))

  attribute_statistics = backend.flat_map(
      eval_report, extract_attribute_statistics, "Extract attribute stat"
  )
  # (i, (org.categorical_statistics, syn.categorical_statistics))

  tv_distances = backend.map_values(
      attribute_statistics,
      lambda xy: _compute_tv_distance(xy[0], xy[1]),
      "ComputeTVDistance",
  )
  # (i, tv_distance)

  tv_distances = backend.to_list(tv_distances, "ToList")
  # [(i, tv_distance)]

  chi2_pvalues = backend.map_values(
      attribute_statistics,
      lambda xy: _compute_chi2_pvalue(dict(xy[0]), dict(xy[1])),
      "ComputeChi2PValues",
  )
  # (i, chi2_pvalue)
  chi2_pvalues = backend.to_list(chi2_pvalues, "ToList")
  # [(i, chi2_pvalue)]

  def add_to_eval_report(
      eval_report: diagnostic_info.TabularEvalReport,
      tv_distances: list[tuple[int, float]],
      chi2_pvalues: list[tuple[int, float]],
  ) -> diagnostic_info.TabularEvalReport:
    tv_dict = dict(tv_distances)
    chi2_dict = dict(chi2_pvalues)

    attribute_eval_reports = []
    for i, attribute_statistics in enumerate(
        eval_report.original_dataset_statistics.attribute_statistics
    ):
      tv_distance = tv_dict.get(i, None)
      chi2_pvalue = chi2_dict.get(i, None)
      report = diagnostic_info.AttributeEvalReport(
          attribute_name=attribute_statistics.attribute_name
      )
      if tv_distance is not None:
        report.tv_distance = tv_distance
      if chi2_pvalue is not None:
        report.chi2_pvalue = chi2_pvalue
      attribute_eval_reports.append(report)
    new_report = dataclasses.replace(
        eval_report, attribute_eval_reports=attribute_eval_reports
    )

    return new_report

  return backend.map_with_side_inputs(
      eval_report,
      add_to_eval_report,
      [tv_distances, chi2_pvalues],
      "AddToEvalReport",
  )


def _compute_tv_distance(
    original_category_value_counts: list[tuple[Any, int]],
    synthetic_category_value_counts: list[tuple[Any, int]],
) -> float:
  """Computes the Total Variation (TV) distance between two distributions."""

  original_counts = dict(original_category_value_counts)
  synthetic_counts = dict(synthetic_category_value_counts)
  n_original = sum(original_counts.values())
  n_synthetic = sum(synthetic_counts.values())

  if n_original == 0 and n_synthetic == 0:
    return 0.0
  if n_original == 0 or n_synthetic == 0:
    return 1.0

  categories = set(original_counts.keys()) | set(synthetic_counts.keys())
  tv_distance = 0.0
  for category in categories:
    p_i = original_counts.get(category, 0) / n_original
    q_i = synthetic_counts.get(category, 0) / n_synthetic
    tv_distance += abs(p_i - q_i)

  return 0.5 * tv_distance


def _compute_chi2_pvalue(sample1: dict[Any, int], sample2: dict[Any, int]):
  """Computes the Chi-squared test p-value between two categorical samples."""
  n1 = sum(sample1.values())
  n2 = sum(sample2.values())
  if n1 == 0 and n2 == 0:
    return 1.0
  if n1 == 0 or n2 == 0:
    return 0.0
  # 1. Get the union of all keys to align the histograms
  all_keys = list(set(sample1.keys()) | set(sample2.keys()))

  # 2. Create the contingency table (2 rows, N columns)
  # Fill missing keys with 0
  row1 = [sample1.get(k, 0) for k in all_keys]
  row2 = [sample2.get(k, 0) for k in all_keys]

  contingency_table = np.array([row1, row2])

  # 3. Compute Chi-squared test
  # stat: The test statistic
  # p_val: The p-value we are looking for
  # dof: Degrees of freedom
  # expected: The expected frequencies based on marginal totals
  _, p_val, _, _ = stats.chi2_contingency(contingency_table)

  return p_val

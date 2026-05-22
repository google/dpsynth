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

"""Computes multi-way marginal total variation distances."""


from collections.abc import Iterable
import itertools
from typing import Any

from dpsynth.eval import types
from dpsynth.pipeline_transformations import diagnostic_info
import pipeline_dp


def compute_multi_way_marginal_distance(
    original_data: types.Collection[types.Record],
    synthetic_data: types.Collection[types.Record],
    categorical_indices: list[int],
    original_size: types.Collection[int],
    synthetic_size: types.Collection[int],
    backend: pipeline_dp.PipelineBackend,
    max_marginal_degree: int = 2,
) -> types.Collection[diagnostic_info.CategoricalMarginal]:
  """Computes TV distance for categorical marginals up to max_marginal_degree.

  Args:
    original_data: Collection of original records.
    synthetic_data: Collection of synthetic records.
    categorical_indices: Indices of categorical attributes.
    original_size: Singleton collection containing original dataset size.
    synthetic_size: Singleton collection containing synthetic dataset size.
    backend: PipelineBackend to use for computations.
    max_marginal_degree: The maximum degree of marginals to compute. Default is
      2.

  Returns:
    A collection of CategoricalMarginal protos.
  """

  def extract_combinations_keys(
      record: types.Record,
  ) -> Iterable[tuple[tuple[int, ...], tuple[Any, ...]]]:
    for r in range(1, max_marginal_degree + 1):
      for indices in itertools.combinations(categorical_indices, r):
        values = tuple(record[i] for i in indices)
        yield (indices, values)

  # ( (indices, values), count )
  org_counts = backend.count_per_element(
      backend.flat_map(original_data, extract_combinations_keys, "OrgExtract"),
      "OrgCount",
  )
  # ((indices, values), count)
  syn_counts = backend.count_per_element(
      backend.flat_map(synthetic_data, extract_combinations_keys, "SynExtract"),
      "SynCount",
  )
  # ((indices, values), count)

  # Join by (indices, values) using raw counts
  org_counts_tagged = backend.map_tuple(
      org_counts, lambda iv, c: (iv, (0, c)), "OrgTag"
  )
  syn_counts_tagged = backend.map_tuple(
      syn_counts, lambda iv, c: (iv, (1, c)), "SynTag"
  )

  all_counts = backend.flatten(
      [org_counts_tagged, syn_counts_tagged], "FlattenCounts"
  )
  # ((indices, values), (tag, count))
  grouped_counts = backend.group_by_key(all_counts, "GroupCounts")
  # ((indices, values), [(tag, count)])

  def compute_metrics(
      grouped_item: tuple[
          tuple[tuple[int, ...], tuple[Any, ...]], Iterable[tuple[int, int]]
      ],
      orig_n: int,
      syn_n: int,
  ) -> tuple[
      tuple[int, ...], tuple[float, int, int, list[tuple[int, tuple[Any, ...]]]]
  ]:
    (indices, values), tagged_counts = grouped_item
    orig_c = 0
    syn_c = 0
    for tag, c in tagged_counts:
      if tag == 0:
        orig_c = c
      else:
        syn_c = c

    p = orig_c / max(1, orig_n)
    q = syn_c / max(1, syn_n)
    abs_diff = abs(p - q)

    unseen_combos = 0
    unseen_occurrences = 0
    top_unseen = []

    if orig_c == 0 and syn_c > 0:
      unseen_combos = 1
      unseen_occurrences = syn_c
      top_unseen = [(syn_c, values)]

    return (indices, (abs_diff, unseen_combos, unseen_occurrences, top_unseen))

  metrics_by_indices = backend.map_with_side_inputs(
      grouped_counts,
      compute_metrics,
      [original_size, synthetic_size],
      "ComputeMetrics",
  )
  # (indices, (abs_diff, unseen_combos, unseen_occurrences, top_unseen))

  def reduce_metrics(
      m1: tuple[float, int, int, list[tuple[int, tuple[Any, ...]]]],
      m2: tuple[float, int, int, list[tuple[int, tuple[Any, ...]]]],
  ) -> tuple[float, int, int, list[tuple[int, tuple[Any, ...]]]]:
    abs_diff1, u_combos1, u_occ1, tu1 = m1
    abs_diff2, u_combos2, u_occ2, tu2 = m2

    merged_tu = tu1 + tu2
    merged_tu.sort(key=lambda x: x[0], reverse=True)
    top3_tu = merged_tu[:3]

    return (
        abs_diff1 + abs_diff2,
        u_combos1 + u_combos2,
        u_occ1 + u_occ2,
        top3_tu,
    )

  reduced_metrics = backend.reduce_per_key(
      metrics_by_indices, reduce_metrics, "ReduceMetrics"
  )
  # (indices, (sum_abs_diff, sum_unseen_combos, sum_u_occ, top3_unseen))

  def to_marginal_proto(
      indices: tuple[int, ...],
      metrics: tuple[float, int, int, list[tuple[int, tuple[Any, ...]]]],
  ) -> diagnostic_info.CategoricalMarginal:
    sum_abs_diff, u_combos, u_occ, tu = metrics

    top_unseen_protos = []
    for count, values in tu:
      str_values = [str(v) for v in values]
      top_unseen_protos.append(
          diagnostic_info.UnseenCombination(values=str_values, count=count)
      )

    return diagnostic_info.CategoricalMarginal(
        attribute_indices=list(indices),
        tv_distance=0.5 * sum_abs_diff,
        num_unseen_combinations=u_combos,
        unseen_occurrences=u_occ,
        top_unseen_combinations=top_unseen_protos,
    )

  return backend.map_tuple(
      reduced_metrics, to_marginal_proto, "ToMarginalProto"
  )
  # (CategoricalMarginal, )

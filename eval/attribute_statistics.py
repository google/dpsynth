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

"""Computes statistics per attribute."""

import collections
import dataclasses
from typing import Any

from eval import correlation_computation
from eval import types
from dpsynth.pipeline_transformations import diagnostic_info
import pipeline_dp
from pipeline_dp import pipeline_functions


@dataclasses.dataclass(frozen=True)
class Accumulator:
  """Accumulator for categorical attributes."""

  num_records: int
  num_non_none_values: int


def _create_categorical_statistics_accumulator(
    values: list[Any],
) -> Accumulator:
  """Creates an accumulator for categorical attributes from a list of values."""
  not_none_values = [value for value in values if value is not None]
  return Accumulator(
      num_records=len(values),
      num_non_none_values=len(not_none_values),
  )


class CategoricalStatisticsCombiner(pipeline_dp.combiners.Combiner):
  """Combiner for computing attribute statistics."""

  def create_accumulator(self, values: list[Any]) -> Accumulator:
    return _create_categorical_statistics_accumulator(values)

  def merge_accumulators(
      self, accum1: Accumulator, accum2: Accumulator
  ) -> Accumulator:
    return Accumulator(
        num_records=accum1.num_records + accum2.num_records,
        num_non_none_values=accum1.num_non_none_values
        + accum2.num_non_none_values,
    )

  def compute_metrics(self, accum: Accumulator) -> tuple[int, int]:
    return (
        accum.num_records,
        accum.num_non_none_values,
    )

  def metrics_names(self) -> list[str]:
    return ["CategoricalStatisticsCombiner"]

  def explain_computation(self):
    pass


def _compute_attribute_values_count(
    data: types.Collection[types.Record],
    attribute_indices: list[int],
    backend: pipeline_dp.PipelineBackend,
) -> types.Collection[tuple[int, list[tuple[Any, int]]]]:
  """Computes the count of each value for each attribute."""

  def get_attributes(record: types.Record):
    for i, value in enumerate(record):
      if i in attribute_indices:
        yield (i, value)

  attributes = backend.flat_map(data, get_attributes, "GetAttributesValues")
  # (i_attribute, value)
  attribute_values_count = backend.count_per_element(attributes, "CountValues")
  # ((i_attribute, value), count)

  attribute_values_count = backend.map_tuple(
      attribute_values_count,
      lambda i_value, count: (i_value[0], (i_value[1], count)),
      "RekeyToAttribute",
  )
  # (i_attribute, (value, count))

  attribute_values_count = backend.group_by_key(
      attribute_values_count, "GroupByAttribute"
  )
  # (i_attribute, [(value, count)])

  return backend.map_values(attribute_values_count, list, "ToUniqueElements")


def compute_dataset_statistics(
    data: types.Collection[types.Record],
    config: diagnostic_info.TabularEvalConfig,
    backend: pipeline_dp.PipelineBackend,
) -> tuple[
    types.Collection[diagnostic_info.DatasetStatistics], types.Collection[int]
]:
  """Computes statistics for all categorical attributes in the dataset.

  Args:
    data: Collection of records.
    config: Tabular evaluation configuration.
    backend: PipelineBackend to use for computations.

  Returns:
    A collection containing DatasetStatistics, and collection containing dataset
    size.
  """
  num_attributes = len(config.attributes)
  idx_categorical_attributes = list(range(num_attributes))

  categorical_values_count = _compute_attribute_values_count(
      data,
      idx_categorical_attributes,
      backend,
  )
  # (i, [(value, count)])
  correlations = correlation_computation.compute_correlations(
      data, config, backend
  )
  # [CorrelationValue]

  num_records_col = pipeline_functions.size(
      backend, data, "Count number elements"
  )

  statistics = backend.flatten(
      [
          categorical_values_count,
          num_records_col,
      ],
      "FlattenStatistics",
  )
  # (num_elements,) | (i, [(value, count)])

  statistics = backend.to_list(statistics, "ToList")

  # singleton [(i, [(value, count)]), num_records]

  def to_dataset_statistics(
      aggregates: list[Any],
      correlations: list[diagnostic_info.CorrelationValue],
  ) -> diagnostic_info.DatasetStatistics:
    num_records = [size for size in aggregates if isinstance(size, int)][0]
    categorical_values_count = collections.defaultdict(lambda: None)
    not_nones = [0] * num_attributes
    for elem in aggregates:
      if isinstance(elem, int):
        continue  # size
      i_attribute, stat = elem

      # categorical value count
      categorical_values_count[i_attribute] = stat
      nones = [count for value, count in stat if value is None]
      not_nones[i_attribute] = num_records - sum(
          nones
      )  # num_records might be not known yet

    attribute_statistics = []
    for i_attribute in range(num_attributes):
      value_count = categorical_values_count[i_attribute]
      if value_count is None:
        categorical_stats_proto = None
      else:
        categorical_stats_proto = (
            diagnostic_info.CategoricalAttributeStatistics(
                num_categories=len(value_count)
            )
        )
        for value, count in value_count:
          if value is not None:
            category_str = str(value)
          else:
            category_str = "None"
          categorical_stats_proto.category_counts.append(
              diagnostic_info.CategoryCount(
                  category=category_str, count=count
              )
          )

      attr_stats_proto = diagnostic_info.AttributeStatistics(
          attribute_name=config.attributes[i_attribute],
          num_non_none_values=not_nones[i_attribute],
          categorical_statistics=categorical_stats_proto,
      )

      attribute_statistics.append(attr_stats_proto)

    return diagnostic_info.DatasetStatistics(
        num_records=num_records,
        attribute_statistics=attribute_statistics,
        correlations=correlations,
    )

  dataset_statistics = backend.map_with_side_inputs(
      statistics,
      to_dataset_statistics,
      [correlations],
      "ToAttributeStatistics",
  )
  return dataset_statistics, num_records_col

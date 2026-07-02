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

"""Beam-backed column initializers for DP Synth.

Computes per-column sufficient statistics via Apache Beam PTransforms,
then runs DP mechanisms from ``primitives.py`` directly on the driver.
No dependency on MBI or JAX — only ``numpy``, ``domain``, ``primitives``,
and ``vectorized_transformations`` are imported.  All outputs are pure
NumPy arrays in lightweight dataclasses.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any

import apache_beam as beam
from dpsynth import domain
from dpsynth.local_mode import vectorized_transformations as vtx
import numpy as np

# representation for large pipelines.  Consider named tuples or Beam Rows.
Row = dict[str, Any]


class ColumnType(enum.Enum):
  NUMERICAL = 'numerical'
  CATEGORICAL = 'categorical'
  OPENSET = 'openset'


@dataclasses.dataclass
class BeamColumnResult:
  """Lightweight column result without MBI/JAX dependency."""

  column_type: ColumnType
  categorical_attribute: domain.CategoricalAttribute
  bin_edges: np.ndarray | None = None
  noisy_counts: np.ndarray | None = None
  stddev: float | None = None


@dataclasses.dataclass
class InitSpec:
  """Per-column mechanism + attribute specification (MBI-free)."""

  column_type: ColumnType
  mechanism: Any  # primitives.DPMechanism subclass
  attribute: Any  # domain.*Attribute
  grid_size: int | None = None  # numerical only
  min_count: int = 1  # openset only


def _build_encode_specs(
    init_specs: dict[str, InitSpec],
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, int]]:
  """Derives encoding specs and openset min-counts from InitSpecs."""
  encode_specs = []
  openset_min_counts = {}
  for column, spec in init_specs.items():
    if spec.column_type == ColumnType.NUMERICAL:
      attr = spec.attribute
      lower = attr.min_value
      delta = (attr.exclusive_max_value - lower) / (spec.grid_size - 1)
      encode_specs.append((
          column,
          'numerical',
          {
              'attribute': attr,
              'lower': lower,
              'delta': delta,
          },
      ))
    elif spec.column_type == ColumnType.CATEGORICAL:
      lookup = {str(v): i for i, v in enumerate(spec.attribute.possible_values)}
      encode_specs.append((
          column,
          'categorical',
          {
              'lookup': lookup,
              'default': spec.attribute.out_of_domain_index,
          },
      ))
    elif spec.column_type == ColumnType.OPENSET:
      encode_specs.append((column, 'openset', {}))
      openset_min_counts[column] = spec.min_count
  return encode_specs, openset_min_counts


class _EncodeColumns(beam.DoFn):
  """Encodes each row into (column, key) pairs for all columns at once."""

  def __init__(self, specs: list[tuple[str, str, dict[str, Any]]]):
    # Do all setup in __init__ so that process below is cheaper.
    # We handle all columns at once here to reduce the size of the DAG in Beam.
    super().__init__()
    self._specs = specs

  def process(self, row: Row):
    for column, kind, params in self._specs:
      if kind == 'numerical':
        value = params['attribute'].standardize(row[column])
        if math.isnan(value):
          continue  # clip_to_range=False: standardize returns NaN --> drop.
        value = round((value - params['lower']) / params['delta'])
        yield (column, value)
      elif kind == 'categorical':
        value = params['lookup'].get(str(row[column]), params['default'])
        yield (column, value)
      elif kind == 'openset':
        yield (column, str(row[column]))


def _unpack_count(element):
  """Restructures ((column, key), count) to (column, (key, count))."""
  (col, key), count = element
  return (col, (key, count))


def _filter_openset(element, min_counts):
  """Filters open-set values below min_count, passes others through."""
  col, pairs = element
  min_count = min_counts.get(col)
  if min_count is not None:
    pairs = [(k, c) for k, c in pairs if c >= min_count]
  return (col, pairs)


def _materialize_pairs(col, pairs):
  """Converts GroupByKey's lazy iterator to a concrete list."""
  return (col, list(pairs))


class ComputeSufficientStats(beam.PTransform):
  """Computes per-column sufficient statistics in a single pass."""

  def __init__(self, init_specs: dict[str, InitSpec]):
    super().__init__()
    self._encode_specs, self._openset_min_counts = _build_encode_specs(
        init_specs
    )

  def expand(
      self, rows: beam.PCollection[Row]
  ) -> beam.PCollection[tuple[str, list[tuple[Any, int]]]]:
    return (
        rows
        | 'Encode' >> beam.ParDo(_EncodeColumns(self._encode_specs))
        | 'Count' >> beam.combiners.Count.PerElement()
        | 'Unpack' >> beam.Map(_unpack_count)
        # Aggregate data and materialize on the driver (see module header).
        | 'GroupByColumn' >> beam.GroupByKey()
        | 'ToLists' >> beam.MapTuple(_materialize_pairs)
        | 'FilterOpenSet'
        >> beam.Map(_filter_openset, min_counts=self._openset_min_counts)
    )


def _sparse_to_dense_numerical(sparse, grid_size):
  """Converts sparse (index, count) pairs to a dense histogram array."""
  counts = np.zeros(grid_size, dtype=np.float64)
  for idx, count in sparse:
    counts[idx] = count
  return counts


def _sparse_to_dense_categorical(sparse, size):
  """Converts sparse (index, count) pairs to a dense count vector."""
  counts = np.zeros(size, dtype=np.float64)
  for idx, count in sparse:
    counts[idx] = count
  return counts


def _sparse_to_openset(sparse):
  """Converts sparse (value, count) pairs to parallel arrays."""
  if not sparse:
    return np.array([], dtype=object), np.array([], dtype=np.float64)
  keys, vals = zip(*sparse)
  return np.array(keys), np.array(vals, dtype=np.float64)


def _edges_to_beam_result(raw_edges, attribute):
  """Processes quantile edges into a BeamColumnResult (no MBI)."""
  raw_edges = np.asarray(raw_edges, dtype=float)
  if attribute.dtype == 'int':
    raw_edges = np.floor(raw_edges)
  bin_edges, _ = np.unique(raw_edges, return_counts=True)
  # Remove edges at max_value to avoid degenerate tail bins.
  if len(bin_edges) > 0 and bin_edges[-1] >= attribute.max_value:
    bin_edges = bin_edges[:-1]
  cat_attr = vtx.categorical_attribute_from_edges(bin_edges, attribute)
  return BeamColumnResult(ColumnType.NUMERICAL, cat_attr, bin_edges=bin_edges)


def run_from_summary(
    sparse_stats: dict[str, list[tuple[Any, int]]],
    init_specs: dict[str, InitSpec],
    rng: np.random.Generator,
) -> dict[str, BeamColumnResult]:
  """Runs DP mechanisms via primitives and returns pure NumPy results."""
  results: dict[str, BeamColumnResult] = {}
  for column, spec in init_specs.items():
    sparse = sparse_stats[column]
    if spec.column_type == ColumnType.NUMERICAL:
      counts = _sparse_to_dense_numerical(sparse, spec.grid_size)
      raw_edges = spec.mechanism(rng, counts)
      results[column] = _edges_to_beam_result(raw_edges, spec.attribute)
    elif spec.column_type == ColumnType.CATEGORICAL:
      counts = _sparse_to_dense_categorical(sparse, spec.attribute.size)
      result = spec.mechanism(rng, counts)
      results[column] = BeamColumnResult(
          ColumnType.CATEGORICAL,
          spec.attribute,
          noisy_counts=result.counts,
          stddev=spec.mechanism.sigma,
      )
    elif spec.column_type == ColumnType.OPENSET:
      unique_values, value_counts = _sparse_to_openset(sparse)
      result = spec.mechanism.from_summary(rng, value_counts)
      selected = list(unique_values[result.selected_partitions])
      possible = [spec.attribute.default_value] + selected
      cat_attr = domain.CategoricalAttribute(
          possible_values=possible,
          out_of_domain_index=0,
      )
      results[column] = BeamColumnResult(
          ColumnType.OPENSET,
          cat_attr,
          noisy_counts=result.estimated_counts,
          stddev=spec.mechanism.sigma,
      )
  return results


class BeamInitialize(beam.PTransform):
  """Computes sufficient stats and runs DP initialization."""

  def __init__(self, init_specs: dict[str, InitSpec], rng: np.random.Generator):
    super().__init__()
    self._init_specs = init_specs
    self._rng = rng

  def expand(
      self, rows: beam.PCollection[Row]
  ) -> beam.PCollection[dict[str, BeamColumnResult]]:
    return (
        rows
        | 'Stats' >> ComputeSufficientStats(self._init_specs)
        | 'ToDict' >> beam.combiners.ToDict()
        | 'Initialize'
        >> beam.Map(
            run_from_summary,
            init_specs=self._init_specs,
            rng=self._rng,
        )
    )

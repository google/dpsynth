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
then delegates to the existing initializers' ``from_summary()`` methods
for DP mechanism execution on the driver. The central assumption in this file
is that the data is too large to feasibly materialize in memory on the driver,
but the per-column sufficient statistics can easily fit. The intention is to
use Beam where it is absolutely needed, but quickly delegate to local-mode
implementations as soon as the sufficient statistics are available, creating
a clear separation of concerns. All beam-related logic necessary to use the
local mode variant of DPSynth is contained in this file.
"""

from __future__ import annotations

import math
from typing import Any

import apache_beam as beam
from dpsynth.local_mode import initialization
import numpy as np

# A single row of tabular data: column name -> raw value.
# representation for large pipelines.  Consider supporting named tuples or
# a schema-aware format (e.g. Beam Rows, protos) to reduce per-element overhead.
Row = dict[str, Any]

Initializer = (
    initialization.NumericalInitializer
    | initialization.CategoricalInitializer
    | initialization.OpenSetCategoricalInitializer
)


class _EncodeColumns(beam.DoFn):
  """Encodes each row into (column, key) pairs for all columns at once."""

  def __init__(self, initializers: dict[str, Initializer]):
    # Do all setup in __init__ so that process below is cheaper.
    # We handle all columns at once here to reduce the size of the DAG in Beam.
    super().__init__()
    self._specs: list[tuple[str, str, dict[str, Any]]] = []
    for column, init in initializers.items():
      if isinstance(init, initialization.NumericalInitializer):
        attr = init.attribute
        lower = attr.min_value
        delta = (attr.exclusive_max_value - lower) / (init.grid_size - 1)
        meta = {'attribute': attr, 'lower': lower, 'delta': delta}
        self._specs.append((column, 'numerical', meta))

      elif isinstance(init, initialization.CategoricalInitializer):
        lookup = {
            str(v): i for i, v in enumerate(init.attribute.possible_values)
        }
        meta = {'lookup': lookup, 'default': init.attribute.out_of_domain_index}
        self._specs.append((column, 'categorical', meta))
      elif isinstance(init, initialization.OpenSetCategoricalInitializer):
        self._specs.append((column, 'openset', {}))
      else:
        raise TypeError(f'Unsupported initializer type: {type(init)}')

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
  """Computes per-column sufficient statistics in a single pass.

  Encodes all columns in one ``DoFn``, then counts via a single
  ``Count.PerElement`` and groups by column. The output is a ``PCollection``
  of ``(column_name, sparse_counts_list)`` pairs.

  Attributes:
    initializers: Calibrated initializers keyed by column name.
  """

  def __init__(self, initializers: dict[str, Initializer]):
    super().__init__()
    self._initializers = initializers
    self._openset_min_counts = {
        col: init.min_count
        for col, init in initializers.items()
        if isinstance(init, initialization.OpenSetCategoricalInitializer)
    }

  def expand(
      self, rows: beam.PCollection[Row]
  ) -> beam.PCollection[tuple[str, list[tuple[Any, int]]]]:
    return (
        rows
        | 'Encode' >> beam.ParDo(_EncodeColumns(self._initializers))
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


# into the Beam pipeline, which can increase setup time for each worker.
def run_from_summary(
    sparse_stats: dict[str, list[tuple[Any, int]]],
    initializers: dict[str, Initializer],
    rng: np.random.Generator,
) -> dict[str, initialization.ColumnMeasurement]:
  """Converts materialized sparse stats to ColumnMeasurements on the driver.

  Meant to be called after ``ComputeSufficientStats`` results have been
  materialized (e.g. via ``beam.combiners.ToDict()``).

  Args:
    sparse_stats: Column-keyed dict of sparse (key, count) pair lists, as
      produced by ``ComputeSufficientStats``.
    initializers: Calibrated initializers keyed by column name.
    rng: NumPy random generator for DP noise.

  Returns:
    Per-column ``ColumnMeasurement`` results.
  """
  results: dict[str, initialization.ColumnMeasurement] = {}
  for column, init in initializers.items():
    sparse = sparse_stats[column]
    if isinstance(init, initialization.NumericalInitializer):
      counts = _sparse_to_dense_numerical(sparse, init.grid_size)
      results[column] = init.from_summary(rng, counts)
    elif isinstance(init, initialization.CategoricalInitializer):
      counts = _sparse_to_dense_categorical(sparse, init.attribute.size)
      results[column] = init.from_summary(rng, counts)
    elif isinstance(init, initialization.OpenSetCategoricalInitializer):
      unique_values, value_counts = _sparse_to_openset(sparse)
      results[column] = init.from_summary(rng, unique_values, value_counts)
  return results


class BeamInitialize(beam.PTransform):
  """End-to-end: computes sufficient stats and runs DP initialization.

  Composes ``ComputeSufficientStats`` with sparse-to-dense conversion and
  ``from_summary()`` calls. Produces a singleton ``PCollection`` containing
  one ``dict[str, ColumnMeasurement]`` with all results.

  Attributes:
    initializers: Calibrated initializers keyed by column name.
    rng: NumPy random generator for DP noise.
  """

  def __init__(
      self,
      initializers: dict[str, Initializer],
      rng: np.random.Generator,
  ):
    super().__init__()
    self._initializers = initializers
    self._rng = rng

  def expand(
      self, rows: beam.PCollection[Row]
  ) -> beam.PCollection[dict[str, initialization.ColumnMeasurement]]:
    return (
        rows
        | 'Stats' >> ComputeSufficientStats(self._initializers)
        | 'ToDict' >> beam.combiners.ToDict()
        | 'Initialize'
        # Since all sufficient stats have been computed and materialized on the
        # driver, passing a single rng is fine here.
        >> beam.Map(
            run_from_summary,
            initializers=self._initializers,
            rng=self._rng,
        )
    )

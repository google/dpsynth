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

"""Experimental Beam adapter for local-mode DP Synth.

.. warning:: This module is experimental.

This module provides a lightweight bridge between the local-mode
TabularSynthesizer and Apache Beam, enabling local-mode features to
run on datasets too large to fit in memory. It is *not* a replacement
for a hardened, pipeline-native DP framework such as PipelineDP,
which should be preferred for production pipelines.

This module may serve as a temporary stopgap until there is better
alignment between the pipeline DP implementations and the local-mode
NumPy-based implementations. How it fits within the broader ecosystem
long-term is an open question.

Compared to the pipeline DP approach, this module:

  - Is limited to Apache Beam (no Apache Spark or other runners).
  - Does not go through the hardened privacy-verification path that
    PipelineDP provides, which offers stronger guarantees around DP
    primitive correctness and audited randomness.
  - Requires the full marginal workload to fit on the driver, since
    the discrete mechanism runs locally after Beam materializes the
    marginals.

However, it can be useful when:

  - You want the TabularSynthesizer API (calibrate -> generate) but
    your data lives in a Beam pipeline rather than a DataFrame.
  - You need a feature that is currently only available in local mode
    (e.g. a specific mechanism, constraint, or transformation) and
    want to apply it to large-scale data.
"""

from __future__ import annotations

import math
import pickle
import tempfile
from typing import Any, Protocol, cast

from absl import logging
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from dpsynth import data_generation_v3
from dpsynth import domain
from dpsynth.local_mode import initialization
from dpsynth.local_mode import vectorized_transformations as vtx
import mbi
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


class _SupportsSupportingCliques(Protocol):
  """Structural type for discrete mechanisms exposing a clique workload.

  Every concrete discrete mechanism (MST, AIM, SWIFT, Direct, Independent,
  ...) implements ``supporting_cliques``, but the ``DPMechanism`` base class
  does not declare it. This Protocol lets us call it in a typed way without
  importing or branching on concrete mechanism classes.
  """

  def supporting_cliques(self, data_domain: mbi.Domain) -> list[mbi.Clique]:
    ...


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
        lower, upper, gs = init._grid_spec
        delta = (upper - lower) / (gs - 1)
        meta = dict(attribute=attr, lower=lower, upper=upper, delta=delta)
        self._specs.append((column, 'numerical', meta))

      elif isinstance(init, initialization.CategoricalInitializer):
        meta = {
            'lookup': init.attribute.lookup,
            'default': init.attribute.out_of_domain_index,
        }
        self._specs.append((column, 'categorical', meta))
      elif isinstance(init, initialization.OpenSetCategoricalInitializer):
        self._specs.append((column, 'openset', {}))
      else:
        raise TypeError(f'Unsupported initializer type: {type(init)}')

  def process(self, row: Row):
    for column, kind, params in self._specs:
      value = row.get(column)
      if kind == 'numerical':
        attribute: domain.NumericalAttribute = params['attribute']
        value = attribute.standardize(value)
        if math.isnan(value):
          continue  # clip_to_range=False: standardize returns NaN --> drop.
        index = int(initialization.encode_to_grid(value, **params))
        yield (column, index)
      elif kind == 'categorical':
        index = params['lookup'].get(str(value), params['default'])
        yield (column, index)
      elif kind == 'openset':
        yield (column, str(value))


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


# Stage 1 of the two-pass pipeline: privately learn each column's domain.
# The raw data is too big for one machine, so Beam computes lightweight
# per-column sufficient statistics (numerical histograms, categorical value
# counts) in a distributed pass. These summaries are small, so we gather them
# on the driver and run the DP mechanism there, producing each column's noised,
# integer-encoded domain (a ColumnMeasurement) that stage 2 consumes.
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


class _EncodeAndProject(beam.DoFn):
  """Integer-encodes each row and emits (clique_index, linear_index) pairs."""

  def __init__(
      self,
      column_measurements: dict[str, initialization.ColumnMeasurement],
      domains: dict[str, Any],
      workload: list[mbi.Clique],
  ):
    super().__init__()
    self._cms = column_measurements
    self._domains = domains
    self._clique_meta: list[tuple[int, mbi.Clique, tuple[int, ...]]] = []
    for idx, clique in enumerate(workload):
      shape = tuple(
          int(column_measurements[c].categorical_attribute.size) for c in clique
      )
      self._clique_meta.append((idx, clique, shape))

  def _encode_value(self, col: str, raw_value: Any) -> int:
    """Encodes a single raw value to an integer index."""
    cm = self._cms[col]
    if cm.bin_edges is not None:
      attr = self._domains[col]
      value = attr.standardize(raw_value)
      if math.isnan(value):
        return 0  # OOD bucket (clip_to_range=False).
      offset = 0 if attr.clip_to_range else 1
      return int(np.searchsorted(cm.bin_edges, value, side='left')) + offset
    else:
      cat = cm.categorical_attribute
      return cat.lookup.get(str(raw_value), cat.out_of_domain_index)

  def process(self, row: Row):
    encoded = {col: self._encode_value(col, row.get(col)) for col in self._cms}
    for clique_idx, clique_cols, shape in self._clique_meta:
      multi_index = tuple(encoded[c] for c in clique_cols)
      linear = int(np.ravel_multi_index(multi_index, shape))
      yield clique_idx, linear


def _unpack_marginal_count(element):
  """Restructures ((clique_idx, linear_idx), count) for GroupByKey."""
  (clique_idx, linear_idx), count = element
  return clique_idx, (linear_idx, count)


def _assemble_dense_marginal(element, clique_meta, mbi_domain):
  """Converts sparse counts to an mbi.Factor for one clique."""
  clique_idx, sparse_pairs = element
  _, clique_cols, shape = clique_meta[clique_idx]
  total_size = math.prod(shape)
  dense = np.zeros(total_size, dtype=np.float64)
  for linear_idx, count in sparse_pairs:
    dense[linear_idx] = count
  return mbi.Factor(mbi_domain.project(clique_cols), dense.reshape(shape))


def _build_mbi_domain(column_measurements):
  """Builds an mbi.Domain from ColumnMeasurement results."""
  attrs = tuple(column_measurements.keys())
  shape = tuple(
      r.categorical_attribute.size for r in column_measurements.values()
  )
  labels = tuple(
      tuple(r.categorical_attribute.possible_values)
      for r in column_measurements.values()
  )
  return mbi.Domain(attributes=attrs, shape=shape, labels=labels)


# Stage 2 of the two-pass pipeline: compute the joint marginals the DP mechanism
# needs. Using the domains from stage 1, Beam integer-encodes each row and, for
# every requested clique (a small set of columns), counts how many rows fall in
# each cell of that clique's joint histogram, summing across the whole dataset
# to build a single mbi.CliqueVector. These counts are exact/non-private: DP
# noise is added later on the driver by the discrete mechanism.
class ComputeMarginals(beam.PTransform):
  """Computes a workload of marginals over integer-encoded rows.

  Takes raw rows plus the ``ColumnMeasurement`` results from stage 1,
  integer-encodes each row, and computes the contingency table for each
  clique in the workload. The output is a singleton ``PCollection``
  containing one ``mbi.CliqueVector``.

  Attributes:
    column_measurements: Per-column results from stage 1 initialization.
    domains: Original attribute domain specs (needed for numerical encoding).
    workload: List of cliques (tuples of column names) to measure.
  """

  def __init__(
      self,
      column_measurements: dict[str, initialization.ColumnMeasurement],
      domains: dict[str, Any],
      workload: list[mbi.Clique],
  ):
    super().__init__()
    self._column_measurements = column_measurements
    self._domains = domains
    self._workload = workload
    self._mbi_domain = _build_mbi_domain(column_measurements)
    self._clique_meta = []
    for idx, clique in enumerate(workload):
      shape = self._mbi_domain.project(clique).shape
      self._clique_meta.append((idx, clique, shape))

  def expand(self, rows: beam.PCollection[Row]):
    mbi_domain = self._mbi_domain

    def _to_clique_vector(factors):
      cliques = tuple(f.domain.attributes for f in factors)
      tables = {cl: f for cl, f in zip(cliques, factors)}
      return mbi.CliqueVector(mbi_domain, cliques, tables)

    return (
        rows
        | 'EncodeProject'
        >> beam.ParDo(
            _EncodeAndProject(
                self._column_measurements, self._domains, self._workload
            )
        )
        | 'CountPerElement' >> beam.combiners.Count.PerElement()
        | 'Unpack' >> beam.Map(_unpack_marginal_count)
        | 'GroupByClique' >> beam.GroupByKey()
        | 'ToLists' >> beam.MapTuple(_materialize_pairs)
        | 'ToFactor'
        >> beam.Map(
            _assemble_dense_marginal,
            clique_meta=self._clique_meta,
            mbi_domain=mbi_domain,
        )
        | 'ToList' >> beam.combiners.ToList()
        | 'BuildCliqueVector' >> beam.Map(_to_clique_vector)
    )


# ---------------------------------------------------------------------------
# End-to-end synthesis: two-pass Beam pipeline + local discrete mechanism.
# ---------------------------------------------------------------------------


def _write_singleton(value: Any, path: str) -> None:
  """Pickles a single driver-bound pipeline result to ``path``."""
  # Used to move small results (column measurements, row count, clique vector)
  # off remote workers. Unlike a module-level collector, writing to a
  # (possibly distributed) filesystem lets the driver read the value back after
  # the pipeline finishes, so this works on distributed runners.
  with FileSystems.create(path) as f:
    f.write(pickle.dumps(value))


def _read_singleton(path: str) -> Any:
  """Reads a pickled result written by ``_write_singleton`` on the driver."""
  with FileSystems.open(path) as f:
    # Trusted input: only ever reads data this pipeline itself wrote.
    return pickle.loads(f.read())  # pylint: disable=g-unsafe-pickle-load


def generate_from_marginals(
    synth: data_generation_v3.TabularSynthesizer,
    rng: np.random.Generator,
    column_measurements: dict[str, initialization.ColumnMeasurement],
    marginals: mbi.CliqueVector,
    total_measurement: mbi.LinearMeasurement,
) -> data_generation_v3.DataGenerationResult:
  """Runs the discrete mechanism and decoding from pre-computed marginals.

  Args:
    synth: A calibrated TabularSynthesizer.
    rng: NumPy random generator for the discrete mechanism's DP noise.
    column_measurements: Per-column results from pass 1 initialization.
    marginals: The exact joint marginals computed by pass 2.
    total_measurement: The DP-noised total-count measurement (clique ``()``).

  Returns:
    A DataGenerationResult containing the synthetic DataFrame.
  """
  # Mirror the local TabularSynthesizer: the discrete mechanism receives the
  # noisy total (clique ``()``) followed by the one-way column measurements as
  # its initial measurements, so it does not re-spend budget measuring them.
  initial_measurements = [total_measurement] + [
      cm.measurement
      for cm in column_measurements.values()
      if cm.measurement is not None
  ]

  mbi_constraints = tuple(c.to_mbi() for c in synth.cross_attribute_constraints)
  logging.info('[DPSynth/Beam]: Running discrete mechanism.')
  result = synth.discrete_mechanism(
      rng,
      data=marginals,
      initial_measurements=initial_measurements,
      constraints=mbi_constraints,
  )
  logging.info('[DPSynth/Beam]: Generated discrete synthetic data.')

  synthetic_columns = {}
  for col, cm in column_measurements.items():
    col_data = result.synthetic_data.to_dict()[col]
    if cm.bin_edges is not None:
      synthetic_columns[col] = vtx.undiscretize(
          col_data,
          cm.bin_edges,
          synth.domains[col],
          rng=rng,
      )
    else:
      synthetic_columns[col] = vtx.discrete_decode(
          col_data,
          cm.categorical_attribute,
      )
  logging.info('[DPSynth/Beam]: Decoded synthetic data.')

  import pandas as pd  # pylint: disable=g-import-not-at-top

  # Emit columns in domain-declaration order for deterministic output.
  column_order = [c for c in synth.domains if c in synthetic_columns]
  return data_generation_v3.DataGenerationResult(
      synthetic_data=pd.DataFrame(synthetic_columns)[column_order],
      discrete_mechanism_result=result,
  )


def run_two_pass(
    synth: data_generation_v3.TabularSynthesizer,
    rng: np.random.Generator,
    create_rows_fn,
    *,
    temp_location: str | None = None,
    pipeline_options=None,
) -> data_generation_v3.DataGenerationResult:
  """Runs the full two-pass Beam pipeline for DP synthetic data generation.

  Pass 1: Computes per-column sufficient statistics and the total row count,
  then runs DP initialization on the driver.
  Pass 2: Integer-encodes rows and computes the marginal workload required by
  the configured discrete mechanism.
  Finally, runs the discrete mechanism and decoding locally on the driver.

  Args:
    synth: A calibrated TabularSynthesizer.
    rng: NumPy random generator.
    create_rows_fn: A callable that takes a beam.Pipeline and returns a
      PCollection[Row]. Called twice (once per pass).
    temp_location: Directory used to pass small singleton results (column
      measurements, row count, clique vector) from the pipeline back to the
      driver. Must be readable and writable by all workers -- i.e. a shared
      distributed filesystem when using a distributed runner. Defaults to a
      local temp directory, which is only valid for in-process runners (e.g. the
      local DirectRunner).
    pipeline_options: Optional Beam pipeline options.

  Returns:
    A DataGenerationResult containing the synthetic DataFrame.

  Raises:
    ValueError: If the synthesizer has not been calibrated.
  """
  total_count_mechanism = synth.total_count_mechanism
  if synth.initializers is None or total_count_mechanism is None:
    raise ValueError('TabularSynthesizer must be calibrated.')
  sigma = total_count_mechanism.sigma
  if sigma is None:
    raise ValueError('TabularSynthesizer must be calibrated.')

  inits: dict[str, Initializer] = synth.initializers  # type: ignore

  temp_dir = temp_location or tempfile.mkdtemp(prefix='dpsynth_beam_')
  cms_path = FileSystems.join(temp_dir, 'column_measurements.pickle')
  count_path = FileSystems.join(temp_dir, 'row_count.pickle')
  marginals_path = FileSystems.join(temp_dir, 'clique_vector.pickle')

  # Pass 1: per-column initialization plus the total row count. Both are
  # written to ``temp_dir`` and read back on the driver.
  with beam.Pipeline(options=pipeline_options) as p:
    rows = create_rows_fn(p)
    cms = rows | BeamInitialize(inits, rng)
    _ = cms | 'WriteColumnMeasurements' >> beam.Map(
        _write_singleton, path=cms_path
    )
    count = rows | 'CountRows' >> beam.combiners.Count.Globally()
    _ = count | 'WriteRowCount' >> beam.Map(_write_singleton, path=count_path)
  column_measurements = _read_singleton(cms_path)
  num_rows = _read_singleton(count_path)
  logging.info('[DPSynth/Beam]: Pass 1 complete.')

  # Add DP noise to the total row count on the driver, matching the local
  # TabularSynthesizer's DPGaussianCount (noisy count, clamped to >= 1).
  total = max(1.0, num_rows + rng.normal(scale=sigma))
  total_measurement = mbi.LinearMeasurement(np.array([total]), (), stddev=sigma)

  # Ask the configured discrete mechanism which marginals it needs, so the
  # pipeline works for any mechanism (MST, AIM, SWIFT, Direct, Independent).
  mbi_domain = _build_mbi_domain(column_measurements)
  mechanism = cast(_SupportsSupportingCliques, synth.discrete_mechanism)
  workload = mechanism.supporting_cliques(mbi_domain)

  # Pass 2: compute the marginal workload.
  with beam.Pipeline(options=pipeline_options) as p:
    rows = create_rows_fn(p)
    marginals = rows | ComputeMarginals(
        column_measurements,
        dict(synth.domains),
        workload,
    )
    _ = marginals | 'WriteCliqueVector' >> beam.Map(
        _write_singleton, path=marginals_path
    )
  clique_vector = _read_singleton(marginals_path)
  logging.info('[DPSynth/Beam]: Pass 2 complete.')

  return generate_from_marginals(
      synth, rng, column_measurements, clique_vector, total_measurement
  )

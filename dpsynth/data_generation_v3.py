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

"""End-to-end DP synthetic tabular data generation using local mode primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses

from absl import logging
import dp_accounting
from dpsynth import constraints
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.discrete_mechanisms import common as dm_common
from dpsynth.local_mode import initialization
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
import mbi
import numpy as np
import pandas as pd


def _create_initializers(
    domains: Mapping[str, domain.AttributeType],
    numerical_bins: int,
    init_delta: float,
) -> dict[str, primitives.DPMechanism]:
  """Creates per-column initializers from the domain specification.

  Args:
    domains: Mapping from column names to attribute domain specifications.
    numerical_bins: Number of bins for numerical discretization.
    init_delta: Delta for open-set categorical partition selection.

  Returns:
    A dictionary mapping column names to uncalibrated initializer instances.

  Raises:
    ValueError: If a column has an unsupported attribute type.
  """
  initializers = {}
  for col, attr in domains.items():
    if isinstance(attr, domain.NumericalAttribute):
      initializers[col] = initialization.NumericalInitializer(
          name=col, num_partitions=numerical_bins, attribute=attr
      )
    elif isinstance(attr, domain.CategoricalAttribute):
      initializers[col] = initialization.CategoricalInitializer(
          name=col, attribute=attr
      )
    elif isinstance(attr, domain.OpenSetCategoricalAttribute):
      initializers[col] = initialization.OpenSetCategoricalInitializer(
          name=col, attribute=attr, delta=init_delta
      )
    else:
      raise ValueError(
          f'Unsupported attribute type for column {col!r}: {type(attr)}'
      )
  return initializers


def _build_mbi_domain(
    results: Mapping[str, initialization.ColumnMeasurement],
) -> mbi.Domain:
  """Builds an mbi.Domain with labels from per-column measurement results."""
  attrs = tuple(results.keys())
  shape = tuple(r.categorical_attribute.size for r in results.values())
  labels = tuple(
      tuple(r.categorical_attribute.possible_values) for r in results.values()
  )
  return mbi.Domain(attributes=attrs, shape=shape, labels=labels)


@dataclasses.dataclass
class DataGenerationResult:
  """Result of end-to-end DP synthetic data generation."""

  synthetic_data: pd.DataFrame
  discrete_mechanism_result: dm_common.DiscreteMechanismResult


@dataclasses.dataclass
class TabularSynthesizer(primitives.DPMechanism):
  """End-to-end DP synthetic data generation mechanism.

  This mechanism encodes input categorical and numerical data into a discrete
  domain using local mode primitives, runs a discrete mechanism on the
  discretized data, and converts the synthetic output back to the original
  domain.

  Usage::

      synth = TabularSynthesizer(domains=domains)
      calibrated = synth.configure(zcdp_rho=1.0)
      result = calibrated(rng, df)
      synthetic_df = result.synthetic_data

  Attributes:
    domains: Mapping from column names to attribute domain specifications.
    discrete_mechanism: The mechanism to run on the discretized data.
    numerical_bins: Number of bins for numerical attribute discretization.
    init_budget_fraction: Fraction of total zCDP budget allocated to per-column
      initialization (the rest goes to the discrete mechanism).
    initializers: Per-column initializer mechanisms. If None, created
      automatically from ``domains`` during ``configure()``.
    skip_compression: Whether to skip domain compression.
    cross_attribute_constraints: Constraints to enforce on generated data.
  """

  domains: Mapping[str, domain.AttributeType]
  discrete_mechanism: discrete_mechanisms.DiscreteMechanism = dataclasses.field(
      default_factory=discrete_mechanisms.MSTMechanism
  )
  numerical_bins: int = 32
  init_budget_fraction: float = 0.1
  initializers: dict[str, primitives.DPMechanism] | None = None
  total_count_mechanism: primitives.DPGaussianCount | None = None
  cross_attribute_constraints: Sequence[constraints.Constraint] = ()

  def configure(
      self,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
  ) -> TabularSynthesizer:
    """Returns a copy configured with the given privacy budget.

    Splits the budget additively, just as it does for ``zcdp_rho``:

    - ``init_budget_fraction`` of ``zcdp_rho`` goes to per-column initializers
      (split evenly, including a total-count mechanism); the remainder goes to
      the discrete mechanism.
    - ``init_budget_fraction`` of ``delta`` is reserved for open-set partition
      selection (split evenly across open-set columns); the remaining delta is
      unused by pure-zCDP sub-mechanisms.

    When ``calibrate(epsilon, delta)`` is called, the base class binary search
    passes the guarantee delta here. Because the thresholding delta is honestly
    reported in the composite ``dp_event``, the binary search automatically
    ensures the overall (epsilon, delta) guarantee is tight.

    Args:
      zcdp_rho: The zCDP privacy budget.
      delta: Overall approximate DP delta for the mechanism. A fraction
        (``init_budget_fraction``) is allocated to partition selection for
        open-set columns. Must be positive when open-set categorical attributes
        are present.

    Returns:
      A new TabularSynthesizer with calibrated sub-mechanisms.

    Raises:
      ValueError: If open-set attributes exist but delta is 0.
    """
    num_open_set = sum(
        isinstance(attr, domain.OpenSetCategoricalAttribute)
        for attr in self.domains.values()
    )
    if num_open_set > 0 and delta <= 0:
      raise ValueError(
          'delta must be positive when open-set categorical attributes are'
          ' present. It is used for Gaussian partition selection.'
      )
    # Split delta across open-set columns, analogous to splitting zcdp_rho.
    # Under calibrate(), any delta not consumed here is automatically
    # available for the zCDP-to-(epsilon, delta) conversion, so this
    # simple additive split is tight.
    thresholding_delta = self.init_budget_fraction * delta
    per_col_delta = (
        thresholding_delta / num_open_set if num_open_set > 0 else 0.0
    )

    inits = self.initializers or _create_initializers(
        self.domains, self.numerical_bins, per_col_delta
    )
    init_rho = self.init_budget_fraction * zcdp_rho
    # +1 for the DPGaussianCount that always measures the total.
    per_col_rho = init_rho / (len(inits) + 1)
    discrete_rho = zcdp_rho - init_rho

    calibrated_inits = {
        col: init.configure(zcdp_rho=per_col_rho) for col, init in inits.items()
    }
    calibrated_total = primitives.DPGaussianCount().configure(
        zcdp_rho=per_col_rho
    )
    calibrated_discrete = self.discrete_mechanism.configure(
        zcdp_rho=discrete_rho
    )
    return dataclasses.replace(
        self,
        initializers=calibrated_inits,
        discrete_mechanism=calibrated_discrete,
        total_count_mechanism=calibrated_total,
    )

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed DpEvent for all sub-mechanisms.

    Returns:
      A ComposedDpEvent combining all initializer and discrete mechanism events.

    Raises:
      ValueError: If configure() has not been called.
    """
    if self.initializers is None or self.total_count_mechanism is None:
      raise ValueError(
          'Must call configure() or calibrate() before accessing dp_event.'
      )
    events = [init.dp_event for init in self.initializers.values()]
    events.append(self.total_count_mechanism.dp_event)
    events.append(self.discrete_mechanism.dp_event)
    return dp_accounting.ComposedDpEvent(events)

  def __call__(
      self, rng: np.random.Generator, data: pd.DataFrame
  ) -> DataGenerationResult:
    """Generates differentially private synthetic data.

    Args:
      rng: A numpy random number generator.
      data: The dataset to generate synthetic data for. Must contain all columns
        specified in ``domains``.

    Returns:
      A DataGenerationResult containing the synthetic DataFrame.

    Raises:
      ValueError: If configure() has not been called or if required columns are
        missing from the input data.
    """
    if self.initializers is None or self.total_count_mechanism is None:
      raise ValueError(
          'Must call configure() or calibrate() before running the mechanism.'
      )
    for col in self.domains:
      if col not in data.columns:
        raise ValueError(
            f'{col=} not found in dataset. Available: {list(data.columns)}'
        )

    # Phase 1: Per-column initialization.
    # Measure total count first, then run per-column initializers.
    any_col = next(iter(self.domains))
    total = max(1.0, self.total_count_mechanism(rng, data[any_col].values))

    results: dict[str, initialization.ColumnMeasurement] = {}
    for col, init in self.initializers.items():
      if isinstance(init, initialization.NumericalInitializer):
        results[col] = init(rng, data[col].values, estimated_total=total)
      else:
        results[col] = init(rng, data[col].values)

    # Phase 2: Encode data to discrete domain.
    discrete_data = {}
    one_way_measurements = []
    for col, result in results.items():
      if result.bin_edges is not None:
        discrete_data[col] = vtx.discretize(
            data[col].values, result.bin_edges, self.domains[col]
        )
      else:
        discrete_data[col] = vtx.discrete_encode(
            data[col].values, result.categorical_attribute
        )
      if result.measurement is not None:
        one_way_measurements.append(result.measurement)

    mbi_domain = _build_mbi_domain(results)
    discrete = mbi.Dataset(discrete_data, mbi_domain)
    logging.info('[DPSynth]: Finished encoding data.')

    # Phase 3: Run the discrete mechanism.
    mbi_constraints = tuple(
        c.to_mbi() for c in self.cross_attribute_constraints
    )
    mechanism_result = self.discrete_mechanism(
        rng,
        data=discrete,
        initial_measurements=one_way_measurements,
        constraints=mbi_constraints,
    )
    synthetic_data = mechanism_result.synthetic_data
    logging.info('[DPSynth]: Generated discrete synthetic data.')

    # Phase 4: Decode synthetic data back to original domain.
    synthetic_columns = {}
    for col, result in results.items():
      col_data = synthetic_data.to_dict()[col]
      if result.bin_edges is not None:
        synthetic_columns[col] = vtx.undiscretize(
            col_data, result.bin_edges, self.domains[col], rng=rng
        )
      else:
        synthetic_columns[col] = vtx.discrete_decode(
            col_data, result.categorical_attribute
        )
    logging.info('[DPSynth]: Converted data back to original domain.')

    column_order = [col for col in data.columns if col in self.domains]
    return DataGenerationResult(
        synthetic_data=pd.DataFrame(synthetic_columns)[column_order],
        discrete_mechanism_result=mechanism_result,
    )

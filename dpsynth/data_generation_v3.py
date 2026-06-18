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

"""End-to-end DP synthetic data generation using local mode primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses

from absl import logging
import dp_accounting
from dpsynth import constraints
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.local_mode import initialization
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
import mbi
from mbi import estimation as mbi_estimation
import numpy as np
import pandas as pd


def _create_initializers(
    domains: Mapping[str, domain.AttributeType],
    numerical_bins: int,
    init_delta: float,
) -> tuple[
    dict[str, primitives.DPMechanism],
    dict[str, primitives.DPMechanism],
]:
  """Creates per-column initializers from the domain specification.

  Args:
    domains: Mapping from column names to attribute domain specifications.
    numerical_bins: Number of bins for numerical discretization.
    init_delta: Delta for open-set categorical partition selection.

  Returns:
    A (closed_domain_initializers, numerical_initializers) tuple.

  Raises:
    ValueError: If a column has an unsupported attribute type.
  """
  closed_inits = {}
  numerical_inits = {}
  for col, attr in domains.items():
    if isinstance(attr, domain.NumericalAttribute):
      numerical_inits[col] = initialization.NumericalInitializer(
          name=col, num_partitions=numerical_bins, attribute=attr
      )
    elif isinstance(attr, domain.CategoricalAttribute):
      closed_inits[col] = initialization.CategoricalInitializer(
          name=col, attribute=attr
      )
    elif isinstance(attr, domain.OpenSetCategoricalAttribute):
      closed_inits[col] = initialization.OpenSetCategoricalInitializer(
          name=col, attribute=attr, delta=init_delta
      )
    else:
      raise ValueError(
          f'Unsupported attribute type for column {col!r}: {type(attr)}'
      )
  return closed_inits, numerical_inits


@dataclasses.dataclass
class DataGenerationV3(primitives.DPMechanism):
  """End-to-end DP synthetic data generation mechanism.

  This mechanism encodes input categorical and numerical data into a discrete
  domain using local mode primitives, runs a discrete mechanism on the
  discretized data, and converts the synthetic output back to the original
  domain.

  Usage::

      v3 = DataGenerationV3(domains=domains)
      calibrated = v3.calibrate(zcdp_rho=1.0)
      synthetic_df = calibrated(rng, df)

  Attributes:
    domains: Mapping from column names to attribute domain specifications.
    discrete_mechanism: The mechanism to run on the discretized data.
    initializers: Per-column initializer mechanisms. If None, created
      automatically from ``domains`` during ``calibrate()``.
    skip_compression: Whether to skip domain compression.
    cross_attribute_constraints: Constraints to enforce on generated data.
  """

  domains: Mapping[str, domain.AttributeType]
  discrete_mechanism: discrete_mechanisms.DiscreteMechanism = dataclasses.field(
      default_factory=discrete_mechanisms.MSTMechanism
  )
  initializers: dict[str, primitives.DPMechanism] | None = None
  total_count_mechanism: primitives.DPGaussianCount | None = None
  cross_attribute_constraints: Sequence[constraints.Constraint] = ()

  def calibrate(
      self,
      *,
      zcdp_rho: float | None = None,
      epsilon: float | None = None,
      delta: float | None = None,
      numerical_bins: int = 32,
      init_budget_fraction: float = 0.1,
  ) -> DataGenerationV3:
    """Returns a calibrated copy of this mechanism.

    Supports two calibration modes:

    1. **zCDP mode** (``zcdp_rho``): Simple additive budget split between
       initialization and the discrete mechanism. If the domain contains
       open-set categorical attributes, ``delta`` must also be provided and
       is used entirely for partition-selection thresholding.
    2. **Approximate DP mode** (``epsilon`` and ``delta``): Two-stage
       calibration. Reserves ``init_budget_fraction * delta`` for open-set
       thresholding, converts ``(epsilon, remaining_delta)`` to zCDP, then
       uses PLD accounting to find the tightest discrete mechanism budget.

    Args:
      zcdp_rho: The zCDP privacy budget. Mutually exclusive with epsilon.
      epsilon: The epsilon privacy parameter. Must be provided with delta.
      delta: The delta privacy parameter. Required with epsilon, and also
        required with zcdp_rho when open-set categorical attributes exist.
      numerical_bins: Number of bins for numerical discretization.
      init_budget_fraction: Fraction of total budget for initialization.

    Returns:
      A new DataGenerationV3 instance with calibrated sub-mechanisms.

    Raises:
      ValueError: If arguments are invalid or delta is missing when required.
    """
    has_zcdp = zcdp_rho is not None
    has_approx = epsilon is not None
    if has_zcdp == has_approx:
      raise ValueError('Specify exactly one of zcdp_rho or (epsilon, delta).')
    if has_approx and delta is None:
      raise ValueError('delta must be provided when epsilon is specified.')

    num_open_set = sum(
        isinstance(attr, domain.OpenSetCategoricalAttribute)
        for attr in self.domains.values()
    )

    if has_zcdp:
      if num_open_set > 0 and delta is None:
        raise ValueError(
            'delta is required when open-set categorical attributes are'
            ' present.'
        )
      init_delta = delta / num_open_set if num_open_set > 0 else 0.0
      return self._calibrate_zcdp(
          zcdp_rho, numerical_bins, init_delta, init_budget_fraction
      )

    # Approximate DP: reserve init_budget_fraction * delta for thresholding.
    thresholding_delta = init_budget_fraction * delta if num_open_set > 0 else 0
    init_delta = thresholding_delta / num_open_set if num_open_set > 0 else 0.0
    remaining_delta = delta - thresholding_delta
    return self._calibrate_approx_dp(
        epsilon,
        delta,
        remaining_delta,
        numerical_bins,
        init_delta,
        init_budget_fraction,
    )

  def _calibrate_zcdp(
      self, zcdp_rho, numerical_bins, init_delta, init_budget_fraction
  ):
    """Simple additive zCDP budget split."""
    if self.initializers is not None:
      all_inits = dict(self.initializers)
      has_closed_domain = any(
          not isinstance(init, initialization.NumericalInitializer)
          for init in all_inits.values()
      )
    else:
      closed_inits, numerical_inits = _create_initializers(
          self.domains, numerical_bins, init_delta
      )
      all_inits = {**closed_inits, **numerical_inits}
      has_closed_domain = bool(closed_inits)

    has_numerical = any(
        isinstance(init, initialization.NumericalInitializer)
        for init in all_inits.values()
    )
    needs_total_count = has_numerical and not has_closed_domain

    init_rho = init_budget_fraction * zcdp_rho
    # If we need a separate total count and have no closed-domain columns to
    # estimate it from, allocate one extra share of init budget for it.
    num_shares = len(all_inits) + (1 if needs_total_count else 0)
    per_col_rho = init_rho / num_shares
    discrete_rho = zcdp_rho - init_rho

    calibrated_inits = {
        col: init.calibrate(zcdp_rho=per_col_rho)
        for col, init in all_inits.items()
    }
    calibrated_total = (
        primitives.DPGaussianCount().calibrate(zcdp_rho=per_col_rho)
        if needs_total_count
        else None
    )
    calibrated_discrete = self.discrete_mechanism.calibrate(
        zcdp_rho=discrete_rho
    )
    return dataclasses.replace(
        self,
        initializers=calibrated_inits,
        discrete_mechanism=calibrated_discrete,
        total_count_mechanism=calibrated_total,
    )

  def _calibrate_approx_dp(
      self,
      epsilon,
      delta,
      remaining_delta,
      numerical_bins,
      init_delta,
      init_budget_fraction,
  ):
    """Two-stage calibration for (epsilon, delta).

    Stage 1: Convert (epsilon, remaining_delta) to zCDP and calibrate
    initializers with a fraction of that budget. Open-set categoricals emit
    ApproximateDpEvents so the accountant tracks thresholding delta.

    Stage 2: With initializer dp_events now fixed, use PLD accounting to find
    the tightest discrete mechanism budget such that the full composition fits
    within (epsilon, delta).

    Args:
      epsilon: The epsilon privacy parameter.
      delta: The full delta privacy parameter.
      remaining_delta: Delta after reserving thresholding budget.
      numerical_bins: Number of bins for numerical discretization.
      init_delta: Per-column delta for open-set partition selection.
      init_budget_fraction: Fraction of zCDP budget for initialization.

    Returns:
      A new DataGenerationV3 instance with calibrated sub-mechanisms.
    """
    if self.initializers is not None:
      inits = dict(self.initializers)
      has_closed_domain = any(
          not isinstance(init, initialization.NumericalInitializer)
          for init in inits.values()
      )
    else:
      closed_inits, numerical_inits = _create_initializers(
          self.domains, numerical_bins, init_delta
      )
      inits = {**closed_inits, **numerical_inits}
      has_closed_domain = bool(closed_inits)
    has_numerical = any(
        isinstance(init, initialization.NumericalInitializer)
        for init in inits.values()
    )
    needs_total_count = has_numerical and not has_closed_domain
    num_columns = len(inits) + (1 if needs_total_count else 0)

    # Stage 1: Convert (epsilon, remaining_delta) to zCDP and calibrate
    # initializers with init_budget_fraction of that budget.
    total_rho = dp_accounting.calibrate_dp_mechanism(
        make_event_from_param=dp_accounting.ZCDpEvent,
        target_epsilon=epsilon,
        target_delta=remaining_delta,
        make_fresh_accountant=dp_accounting.rdp.RdpAccountant,
    )
    init_rho = init_budget_fraction * total_rho
    per_col_rho = init_rho / num_columns
    calibrated_inits = {
        col: init.calibrate(zcdp_rho=per_col_rho) for col, init in inits.items()
    }
    calibrated_total = (
        primitives.DPGaussianCount().calibrate(zcdp_rho=per_col_rho)
        if needs_total_count
        else None
    )
    # Stage 2: With init dp_events fixed, find the tightest discrete budget.
    # The accountant handles ApproximateDpEvent deltas from open-set
    # initializers automatically.
    init_events = [init.dp_event for init in calibrated_inits.values()]
    if calibrated_total is not None:
      init_events.append(calibrated_total.dp_event)

    # Determine accountant type based on discrete mechanism's dp_event.
    probe_event = self.discrete_mechanism.calibrate(zcdp_rho=1.0).dp_event
    if isinstance(probe_event, dp_accounting.ZCDpEvent):
      make_fresh_accountant = dp_accounting.rdp.RdpAccountant
    else:
      make_fresh_accountant = dp_accounting.pld.PLDAccountant

    def make_event_from_param(discrete_rho):
      discrete_event = self.discrete_mechanism.calibrate(
          zcdp_rho=discrete_rho
      ).dp_event
      return dp_accounting.ComposedDpEvent(init_events + [discrete_event])

    optimal_discrete_rho = dp_accounting.calibrate_dp_mechanism(
        make_event_from_param=make_event_from_param,
        target_epsilon=epsilon,
        target_delta=delta,
        make_fresh_accountant=make_fresh_accountant,
    )

    calibrated_discrete = self.discrete_mechanism.calibrate(
        zcdp_rho=optimal_discrete_rho
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
      ValueError: If calibrate() has not been called.
    """
    if self.initializers is None:
      raise ValueError('Must call calibrate() before accessing dp_event.')
    events = [init.dp_event for init in self.initializers.values()]
    if self.total_count_mechanism is not None:
      events.append(self.total_count_mechanism.dp_event)
    events.append(self.discrete_mechanism.dp_event)
    return dp_accounting.ComposedDpEvent(events)

  def __call__(
      self, rng: np.random.Generator, data: pd.DataFrame
  ) -> pd.DataFrame:
    """Generates differentially private synthetic data.

    Args:
      rng: A numpy random number generator.
      data: The dataset to generate synthetic data for. Must contain all columns
        specified in ``domains``.

    Returns:
      A synthetic DataFrame with the same domain columns as the input.

    Raises:
      ValueError: If calibrate() has not been called or if required columns are
        missing from the input data.
    """
    if self.initializers is None:
      raise ValueError('Must call calibrate() before running the mechanism.')
    for col in self.domains:
      if col not in data.columns:
        raise ValueError(
            f'{col=} not found in dataset. Available: {list(data.columns)}'
        )

    # Phase 1: Per-column initialization.
    # Initialize closed-domain columns first to estimate the total count,
    # then pass it to numerical initializers for heuristic measurements.
    col_results: dict[str, initialization.ColumnMeasurement] = {}
    closed_domain_measurements = []
    for col, init in self.initializers.items():
      if not isinstance(init, initialization.NumericalInitializer):
        col_results[col] = init(rng, data[col].values)
        if col_results[col].measurement is not None:
          closed_domain_measurements.append(col_results[col].measurement)

    # Estimate total from closed-domain measurements or DPGaussianCount.
    estimated_total = None
    if closed_domain_measurements:
      estimated_total = mbi_estimation.minimum_variance_unbiased_total(
          closed_domain_measurements
      )
    elif self.total_count_mechanism is not None:
      # Pick an arbitrary column to count records.
      any_col = next(iter(self.domains))
      estimated_total = max(
          1.0, self.total_count_mechanism(rng, data[any_col].values)
      )

    for col, init in self.initializers.items():
      if isinstance(init, initialization.NumericalInitializer):
        col_results[col] = init(
            rng, data[col].values, estimated_total=estimated_total
        )

    # Phase 2: Encode data to discrete domain.
    discrete_domains = {}
    discrete_data = {}
    one_way_measurements = []
    for col, result in col_results.items():
      discrete_domains[col] = result.categorical_attribute.size
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

    discrete = mbi.Dataset(discrete_data, mbi.Domain.fromdict(discrete_domains))
    logging.info('[DPSynth]: Finished encoding data.')

    # Phase 3: Run the discrete mechanism.
    initial_potentials = constraints.get_initial_parameters(
        self.cross_attribute_constraints, discrete.domain
    )
    model = self.discrete_mechanism(
        rng,
        data=discrete,
        initial_measurements=one_way_measurements,
        initial_potentials=initial_potentials,
    )
    synthetic_data = model.synthetic_data()
    logging.info('[DPSynth]: Generated discrete synthetic data.')

    # Phase 4: Decode synthetic data back to original domain.
    synthetic_columns = {}
    for col, result in col_results.items():
      col_data = synthetic_data.df[col].values
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
    return pd.DataFrame(synthetic_columns)[column_order]

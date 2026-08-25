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

"""Wrapper that adds one-way marginals and domain compression to any mechanism.

``DiscreteConfig`` composes an inner discrete mechanism (AIM, MST,
SWIFT, etc.) with shared pre- and post-processing: one-way marginal
measurement, domain compression, and decompression.  It is the recommended
entry point for purely discrete tables; mixed-type tables should use
``TabularSynthesizer`` in ``data_generation_v3.py`` instead.


Note: This mechanism is not intended to be called directly. It should typically
be used within `DiscreteMechanism` or `TabularSynthesizer`. Users who call it
directly will miss out on features like 1-way measurement selection and domain
compression.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import mst
import mbi
import numpy as np


@dataclasses.dataclass(frozen=True)
class DiscreteConfig(api.MechanismConfig):
  """Wraps an inner mechanism with one-way measurement and compression.

  Attributes:
    mechanism: The inner mechanism config (e.g. ``AIMConfig()``).
    compress_columns: Domain compression config. True = all, list = specific.
    one_way_budget_fraction: Fraction of zCDP budget for one-way marginals.
    constraints: Default MBI constraints to enforce. Can be overridden at call
      time via the ``constraints`` kwarg on ``DiscreteMechanism.__call__``.
  """

  mechanism: api.MechanismConfig = mst.MSTConfig()
  compress_columns: bool | Sequence[str] = False
  one_way_budget_fraction: float = 0.1
  constraints: Sequence[mbi.Constraint] = ()

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Delegates to the inner mechanism's supporting_cliques."""
    if not hasattr(self.mechanism, 'supporting_cliques'):
      raise ValueError('Inner mechanism does not support supporting_cliques.')
    return self.mechanism.supporting_cliques(domain)

  def configure(self, _=None, *, zcdp_rho, delta=0, max_records_per_user=1):
    """Configures the synthesizer with a zCDP budget."""
    api.validate_max_records_per_user(max_records_per_user)

    one_way_rho = zcdp_rho * self.one_way_budget_fraction
    remaining_rho = zcdp_rho - one_way_rho
    inner = self.mechanism.configure(
        _,
        zcdp_rho=remaining_rho,
        delta=delta,
        max_records_per_user=max_records_per_user,
    )
    return DiscreteMechanism(
        config=self,
        inner=inner,
        one_way_gdp_budget=accounting.zcdp_to_gdp(one_way_rho),
        max_records_per_user=max_records_per_user,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiscreteMechanism(api.CalibratedMechanism):
  """Calibrated synthesizer: one-way + compress + inner + decompress."""

  config: DiscreteConfig
  inner: api.CalibratedMechanism
  one_way_gdp_budget: float
  max_records_per_user: int = 1

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Composes one-way measurement event with the inner mechanism's event."""
    events = []
    if self.one_way_gdp_budget > 0:
      noise_multiplier = accounting.gdp_gaussian_sigma(self.one_way_gdp_budget)
      events.append(dp_accounting.GaussianDpEvent(noise_multiplier))

    inner_event = self.inner.dp_event
    if isinstance(inner_event, dp_accounting.ComposedDpEvent):
      events.extend(inner_event.events)
    elif not isinstance(inner_event, dp_accounting.NoOpDpEvent):
      events.append(inner_event)

    # Filter out any lingering NoOpDpEvents from the inner event
    events = [e for e in events if not isinstance(e, dp_accounting.NoOpDpEvent)]

    if not events:
      return dp_accounting.NoOpDpEvent()
    if len(events) == 1:
      return events[0]
    return dp_accounting.ComposedDpEvent(events)

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      constraints: Sequence[mbi.Constraint] | None = None,
  ) -> common.DiscreteMechanismResult:
    """Runs the one-way + compress + inner mechanism + decompress pipeline.

    Args:
      rng: A numpy random number generator.
      data: The dataset to generate synthetic data for.
      initial_measurements: Pre-measured one-way marginals to use instead of
        measuring them here (e.g. fed from TabularSynthesizer).
      constraints: MBI constraints to enforce. If None, falls back to
        ``self.config.constraints``.

    Returns:
      A DiscreteMechanismResult containing the synthetic data and diagnostics.
    """
    if constraints is None:
      constraints = self.config.constraints

    if initial_measurements is not None:
      measurements = list(initial_measurements)
    elif self.one_way_gdp_budget > 0:
      one_way_cliques = [(a,) for a in data.domain]
      if hasattr(data, 'cliques'):
        supported = common.downward_closure(data.cliques)
        one_way_cliques = [cl for cl in one_way_cliques if cl in supported]
      measurements = common.measure_marginals_with_noise(
          rng=rng,
          data=data,  # pyrefly: ignore[bad-argument-type]
          marginal_queries=one_way_cliques,  # pyrefly: ignore[bad-argument-type]
          gdp_sigma=accounting.gdp_gaussian_sigma(self.one_way_gdp_budget),
          max_records_per_user=self.max_records_per_user,
      )
    else:
      measurements = []

    mappings = common.compression_mappings(
        measurements,
        self.config.compress_columns,
        constraints,
    )
    # Compression only supported with mbi.Dataset, not mbi.CliqueVector.
    if mappings and hasattr(data, 'compress'):
      data = data.compress(mappings)  # pyrefly: ignore[bad-argument-type]
      measurements = [m.compress(mappings, data.domain) for m in measurements]  # pyrefly: ignore[bad-argument-type]

    result = self.inner(
        rng,
        data,
        initial_measurements=measurements,
        constraints=constraints,
    )
    if mappings:
      result = dataclasses.replace(
          result,
          synthetic_data=result.synthetic_data.decompress(mappings),
          mappings=mappings,
      )
    return result

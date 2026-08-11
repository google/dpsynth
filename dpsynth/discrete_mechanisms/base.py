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

"""Base classes for the select-measure-estimate paradigm.

This module defines the ``DiscreteSynthesizer`` wrapper class, which handles
one-way measurement and domain compression, and delegates the remaining budget
and functionality to an inner ``mechanism``.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


@dataclasses.dataclass
class DiscreteSynthesizer(api.DPMechanism):
  """Wrapper class that delegates to a sub-mechanism after domain compression.

  This mechanism orchestrates the data preprocessing pipeline::

      check_calibration → measure_one_way → compress → sub_mechanism →
      decompress

  Attributes:
    mechanism: The inner mechanism to delegate to after data preprocessing.
    compress_columns: Domain compression config. True = all, list = specific.
    one_way_budget_fraction: Fraction of zCDP budget for one-way marginals.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Added noise is scaled by this factor.
    zcdp_rho: Total zCDP budget (set by configure).
    one_way_rho: zCDP budget for one-way measurements (set by configure).
  """

  mechanism: common.DiscreteMechanismProtocol
  compress_columns: bool | Sequence[str] = False
  one_way_budget_fraction: float = 1 / 3
  max_records_per_user: int = 1
  zcdp_rho: float | None = None
  one_way_rho: float | None = dataclasses.field(default=None, repr=False)

  def __post_init__(self):
    api.validate_max_records_per_user(self.max_records_per_user)

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    return getattr(self.mechanism, 'supporting_cliques')(domain)

  def configure(
      self,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      **kwargs,
  ) -> DiscreteSynthesizer:
    if initial_measurements is not None or self.one_way_budget_fraction <= 0:
      one_way_rho = None
    else:
      one_way_rho = zcdp_rho * self.one_way_budget_fraction
    remaining_rho = zcdp_rho - (one_way_rho or 0.0)

    configured_sub = self.mechanism.configure(
        zcdp_rho=remaining_rho, delta=delta, **kwargs
    )

    return dataclasses.replace(
        self,
        zcdp_rho=zcdp_rho,
        one_way_rho=one_way_rho,
        mechanism=configured_sub,
    )

  def _one_way_dp_event(self):
    if self.one_way_rho is None:
      return []
    return [
        dp_accounting.GaussianDpEvent(
            noise_multiplier=accounting.zcdp_gaussian_sigma(self.one_way_rho)
        )
    ]

  def _check_calibration(self):
    if self.zcdp_rho is None:
      raise ValueError('Must call calibrate() before using the mechanism.')

  def _one_way_cliques(self, data):
    if hasattr(self.mechanism, 'one_way_cliques'):
      return self.mechanism.one_way_cliques(data)
    cliques = [(a,) for a in data.domain]
    if hasattr(data, 'cliques'):
      supported = common.downward_closure(data.cliques)
      cliques = [cl for cl in cliques if cl in supported]
    return cliques

  def _measure_one_way(
      self, rng, data, phase_times, *, initial_measurements=None
  ):
    if initial_measurements is not None:
      return list(initial_measurements)
    if self.one_way_rho is None:
      return []
    with common.timed(phase_times, 'measurement'):
      sigma = accounting.zcdp_gaussian_sigma(self.one_way_rho)
      cliques = self._one_way_cliques(data)
      return common.measure_marginals_with_noise(
          rng,
          data,
          cliques,
          sigma,
          max_records_per_user=self.max_records_per_user,
      )

  def _compress(self, data, measurements, constraints):
    mappings = common.compression_mappings(
        measurements, self.compress_columns, constraints
    )
    if mappings and hasattr(data, 'compress'):
      data = data.compress(mappings)
      measurements = [m.compress(mappings, data.domain) for m in measurements]
    return data, measurements, mappings

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    return dp_accounting.ComposedDpEvent([
        *self._one_way_dp_event(),
        *([self.mechanism.dp_event] if self.mechanism.dp_event else []),
    ])

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      constraints: Sequence[mbi.Constraint] = (),
  ) -> common.DiscreteMechanismResult:
    self._check_calibration()
    phase_times = {}
    measurements = self._measure_one_way(
        rng, data, phase_times, initial_measurements=initial_measurements
    )
    data, measurements, mappings = self._compress(
        data, measurements, constraints
    )
    result = self.mechanism(
        rng, data, initial_measurements=measurements, constraints=constraints
    )

    # Merge phase times if present.
    if hasattr(result, 'diagnostics') and hasattr(
        result.diagnostics, 'phase_times'
    ):
      for k, v in phase_times.items():
        if k in result.diagnostics.phase_times:
          result.diagnostics.phase_times[k] += v
        else:
          result.diagnostics.phase_times[k] = v

    if mappings:
      result = dataclasses.replace(
          result,
          synthetic_data=result.synthetic_data.decompress(mappings),
          mappings=mappings,
      )
    return result

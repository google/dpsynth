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

"""Implementation of the direct mechanism."""

import dataclasses

import dp_accounting
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
from dpsynth.local_mode import primitives
import mbi
import numpy as np


@dataclasses.dataclass
class DirectMechanism(primitives.DPMechanism):
  """Configuration for the direct mechanism.

  Attributes:
    prespecified_marginal_queries: A list of k-way marginals that a user has
      specified, ONLY these will be used outside of the initial measurements.
    pgm_iters: The number of iterations for the mirror descent algorithm.
    marginal_oracle: The marginal oracle to use for the mirror descent
      algorithm.
    gdp_sigma: The GDP sigma of the end-to-end mechanism. Privacy budget is
      split across the prespecified marginal queries internally.
  """

  prespecified_marginal_queries: list[tuple[str, ...]]
  pgm_iters: int = 5000
  marginal_oracle: mbi.MarginalOracle | None = None
  gdp_sigma: float | None = None

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the prespecified marginal queries."""
    del domain  # Unused; cliques are user-specified.
    return list(self.prespecified_marginal_queries)

  def calibrate(self, *, zcdp_rho: float) -> 'DirectMechanism':
    """Returns a copy calibrated to the given zCDP budget."""
    return dataclasses.replace(
        self, gdp_sigma=accounting.zcdp_gaussian_sigma(zcdp_rho)
    )

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the direct mechanism."""
    if self.gdp_sigma is None:
      raise ValueError('Must call calibrate() before using the mechanism.')
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.gdp_sigma)

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Projectable,
      *,
      initial_measurements: list[mbi.LinearMeasurement] | None = None,
      initial_potentials: mbi.CliqueVector | None = None,
  ) -> common.DiscreteMechanismResult:
    """Generate synthetic data using user specified two way marginals."""
    if self.gdp_sigma is None:
      raise ValueError('Must call calibrate() before using the mechanism.')

    phase_times = {}

    # measure_marginals_with_noise splits gdp_sigma across the queries
    # internally via weight normalization.
    with common.timed(phase_times, 'measurement'):
      new_measurements = common.measure_marginals_with_noise(
          rng, data, self.prespecified_marginal_queries, self.gdp_sigma
      )
      if initial_measurements:
        all_measurements = initial_measurements + new_measurements
      else:
        all_measurements = new_measurements

    # fit a distribution to the noisy measurements
    with common.timed(phase_times, 'estimation'):
      model = mbi.estimation.MirrorDescent(
          marginal_oracle=self.marginal_oracle,
      ).estimate(
          data.domain,
          all_measurements,
          iters=self.pgm_iters,
          potentials=initial_potentials,
      )
    diagnostics = common.clique_stats(model)
    diagnostics.phase_times = phase_times
    return common.DiscreteMechanismResult(
        model=model,
        synthetic_data=model.synthetic_data(),
        measurements=all_measurements,
        diagnostics=diagnostics,
    )

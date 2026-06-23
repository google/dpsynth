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

"""This mechanisms measures all 1-way marginals via the Gaussian mechanism."""

import dataclasses

import dp_accounting
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
from dpsynth.local_mode import primitives
import mbi
import numpy as np


@dataclasses.dataclass
class IndependentMechanism(primitives.DPMechanism):
  """Configuration for the independent mechanism.

  Attributes:
    pgm_iters: The number of iterations for the mirror descent algorithm.
    marginal_oracle: The marginal oracle to use for the mirror descent
      algorithm.
    gdp_sigma: The GDP sigma of the end-to-end mechanism. Privacy budget is
      split across the one-way marginals internally.
  """

  pgm_iters: int = 5000
  marginal_oracle: mbi.MarginalOracle | None = None
  gdp_sigma: float | None = None

  def calibrate(self, *, zcdp_rho: float) -> 'IndependentMechanism':
    """Returns a copy calibrated to the given zCDP budget."""
    return dataclasses.replace(
        self, gdp_sigma=accounting.zcdp_gaussian_sigma(zcdp_rho)
    )

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the independent mechanism."""
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
    """Generate synthetic data via the independent mechanism."""
    if self.gdp_sigma is None:
      raise ValueError('Must call calibrate() before using the mechanism.')
    constraints = initial_potentials is not None
    marginal_oracle = common.default_oracle(self.marginal_oracle, constraints)

    # Split end-to-end gdp_sigma across the d one-way marginals:
    # per-query sigma = gdp_sigma * sqrt(d).
    attributes = len(data.domain)
    per_query_sigma = self.gdp_sigma * attributes**0.5
    measurements = initial_measurements or []
    existing_cliques = {m.clique for m in measurements}
    for attr in data.domain:
      clique = (attr,)
      if clique in existing_cliques:
        continue
      marginal = data.project(clique).datavector()
      noisy_marginal = (
          marginal + rng.normal(size=marginal.shape) * per_query_sigma
      )
      measurements.append(mbi.LinearMeasurement(noisy_marginal, clique))

    potentials = initial_potentials
    if potentials is not None:
      potentials = potentials.expand([m.clique for m in measurements])

    model = mbi.estimation.mirror_descent(
        data.domain,
        measurements,
        iters=self.pgm_iters,
        potentials=potentials,
        marginal_oracle=marginal_oracle,
    )
    return common.DiscreteMechanismResult(
        model=model, measurements=measurements
    )

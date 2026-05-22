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
import mbi
import numpy as np


@dataclasses.dataclass
class DirectConfig:
  """Configuration for the direct mechanism.

  Attributes:
    prespecified_marginal_queries: A list of k-way marginals that a user has
      specified, ONLY these will be used outside of the initial measurements.
    seed: The seed for the random number generator.
    pgm_iters: The number of iterations for the mirror descent algorithm.
    marginal_oracle: The marginal oracle to use for the mirror descent
      algorithm.
  """

  prespecified_marginal_queries: list[tuple[str, ...]]
  seed: int = 0
  pgm_iters: int = 5000
  marginal_oracle: mbi.MarginalOracle | None = None

  def dp_event(self, zcdp_rho: float) -> dp_accounting.DpEvent:
    """Returns the DP event for the direct mechanism."""
    sigma = accounting.zcdp_gaussian_sigma(zcdp_rho)
    return dp_accounting.GaussianDpEvent(noise_multiplier=sigma)


def run_mechanism(
    data: mbi.Projectable,
    config: DirectConfig,
    zcdp_rho: float,
    *,
    initial_measurements: list[mbi.LinearMeasurement] | None = None,
    initial_potentials: mbi.CliqueVector | None = None,
) -> mbi.MarkovRandomField:
  """Generate synthetic data using user specified two way marginals."""
  constraints = initial_potentials is not None
  marginal_oracle = common.default_oracle(config.marginal_oracle, constraints)

  np.random.seed(config.seed)
  # the entire remaining budget rho can be used for measuring the
  # provided marginals with the gauss mechanism - no
  # budget spent on selection
  gdp_sigma = accounting.zcdp_gaussian_sigma(zcdp_rho)

  new_measurements = common.measure_marginals_with_noise(
      data, config.prespecified_marginal_queries, gdp_sigma
  )
  if initial_measurements:
    all_measurements = initial_measurements + new_measurements
  else:
    all_measurements = new_measurements

  # fit a distribution to the noisy measurements
  model = mbi.estimation.mirror_descent(
      data.domain,
      all_measurements,
      iters=config.pgm_iters,
      potentials=initial_potentials,
      marginal_oracle=marginal_oracle,
  )
  return model

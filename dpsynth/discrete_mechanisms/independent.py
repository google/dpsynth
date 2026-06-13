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
import jax
import mbi


@dataclasses.dataclass
class IndependentConfig:
  """Configuration for the independent mechanism.

  Attributes:
    pgm_iters: The number of iterations for the mirror descent algorithm.
    seed: The seed for the random number generator.
    marginal_oracle: The marginal oracle to use for the mirror descent
      algorithm.
  """

  pgm_iters: int = 5000
  seed: int = 0
  marginal_oracle: mbi.MarginalOracle | None = None

  def dp_event(self, zcdp_rho: float) -> dp_accounting.DpEvent:
    """Returns the DP event for the independent mechanism."""
    sigma = accounting.zcdp_gaussian_sigma(zcdp_rho)
    return dp_accounting.GaussianDpEvent(noise_multiplier=sigma)


def run_mechanism(
    data: mbi.Projectable,
    config: IndependentConfig,
    zcdp_rho: float,
    *,
    initial_measurements: list[mbi.LinearMeasurement] | None = None,
    initial_potentials: mbi.CliqueVector | None = None,
) -> mbi.MarkovRandomField:
  """Generate synthetic data via the independent mechanism."""
  constraints = initial_potentials is not None
  marginal_oracle = common.default_oracle(config.marginal_oracle, constraints)

  gdp_budget = accounting.zcdp_to_gdp(zcdp_rho)

  attributes = len(data.domain)
  sigma = accounting.gdp_gaussian_sigma(gdp_budget / attributes)
  measurements = initial_measurements or []
  keys = jax.random.split(jax.random.key(config.seed), attributes)
  for attr, key in zip(data.domain, keys):
    clique = (attr,)
    marginal = data.project(clique).datavector()
    noisy_marginal = marginal + jax.random.normal(key, marginal.shape) * sigma
    measurements.append(mbi.LinearMeasurement(noisy_marginal, clique))

  potentials = initial_potentials
  if potentials is not None:
    # `measurements` can contain the same clique more than once (the one-way
    # marginals passed in via `initial_measurements` plus the one-way marginals
    # measured in the loop above). `CliqueVector.expand` requires unique cliques
    # and otherwise raises "Cliques must be unique.", so de-duplicate while
    # preserving order before expanding.
    unique_cliques = list(dict.fromkeys(m.clique for m in measurements))
    potentials = potentials.expand(unique_cliques)

  model = mbi.estimation.mirror_descent(
      data.domain,
      measurements,
      iters=config.pgm_iters,
      potentials=potentials,
      marginal_oracle=marginal_oracle,
  )
  return model

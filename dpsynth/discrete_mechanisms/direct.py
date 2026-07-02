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

from collections.abc import Sequence
import dataclasses

from absl import logging
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
    compress_columns: Controls domain compression. True compresses all columns,
      False disables compression, or a list of column names to compress.
    pgm_iters: The number of iterations for the mirror descent algorithm.
    marginal_oracle: The marginal oracle to use for the mirror descent
      algorithm.
    gdp_sigma: The GDP sigma of the end-to-end mechanism. Privacy budget is
      split across the prespecified marginal queries internally.
  """

  prespecified_marginal_queries: list[tuple[str, ...]]
  compress_columns: bool | Sequence[str] = False
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
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: list[mbi.LinearMeasurement] | None = None,
      constraints: tuple[mbi.Constraint, ...] = (),
  ) -> common.DiscreteMechanismResult:
    """Generate synthetic data using user specified two way marginals."""
    if self.gdp_sigma is None:
      raise ValueError('Must call calibrate() before using the mechanism.')

    phase_times = {}

    # Domain compression: merge rare values to shrink the state space.
    one_way = (
        [m for m in initial_measurements if len(m.clique) == 1]
        if initial_measurements
        else []
    )
    mappings = common.compression_mappings(
        one_way, self.compress_columns, constraints
    )
    if mappings:
      data = data.compress(mappings)
    if mappings and initial_measurements:
      initial_measurements = [
          m.compress(mappings, data.domain) for m in initial_measurements
      ]

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

    logging.info(
        '[Direct]:\n%s',
        mbi.summarize(data.domain, [m.clique for m in all_measurements]),
    )
    # fit a distribution to the noisy measurements
    with common.timed(phase_times, 'estimation'):
      estimator = mbi.estimation.MirrorDescent(self.marginal_oracle)
      model = estimator.estimate(
          data.domain,
          all_measurements,
          iters=self.pgm_iters,
          constraints=constraints,
      )
    diagnostics = common.clique_stats(model)
    diagnostics.phase_times = phase_times
    synthetic_data = model.synthetic_data()
    if mappings:
      synthetic_data = synthetic_data.decompress(mappings)
    return common.DiscreteMechanismResult(
        model=model,
        synthetic_data=synthetic_data,
        measurements=all_measurements,
        diagnostics=diagnostics,
        mappings=mappings,
    )

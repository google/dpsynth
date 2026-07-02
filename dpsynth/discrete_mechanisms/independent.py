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

from collections.abc import Sequence
import dataclasses

from absl import logging
import dp_accounting
from dpsynth import _api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


@dataclasses.dataclass
class IndependentMechanism(_api.DPMechanism):
  """Configuration for the independent mechanism.

  Attributes:
    pgm_iters: The number of iterations for the mirror descent algorithm.
    compress_columns: Controls domain compression. True compresses all columns,
      False disables compression, or a list of column names to compress.
    marginal_oracle: The marginal oracle to use for the mirror descent
      algorithm.
    gdp_sigma: The GDP sigma of the end-to-end mechanism. Privacy budget is
      split across the one-way marginals internally.
  """

  pgm_iters: int = 5000
  compress_columns: bool | Sequence[str] = False
  marginal_oracle: mbi.MarginalOracle | None = None
  gdp_sigma: float | None = None

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the one-way marginals this mechanism will measure."""
    return [(a,) for a in domain.attributes]

  def configure(
      self, *, zcdp_rho: float, delta: float = 0.0
  ) -> 'IndependentMechanism':
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
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: list[mbi.LinearMeasurement] | None = None,
      constraints: tuple[mbi.Constraint, ...] = (),
  ) -> common.DiscreteMechanismResult:
    """Generate synthetic data via the independent mechanism."""
    if self.gdp_sigma is None:
      raise ValueError('Must call calibrate() before using the mechanism.')

    # Split end-to-end gdp_sigma across the d one-way marginals:
    # per-query sigma = gdp_sigma * sqrt(d).
    attributes = len(data.domain)
    per_query_sigma = self.gdp_sigma * attributes**0.5
    phase_times = {}
    measurements = initial_measurements or []
    existing_cliques = {m.clique for m in measurements}
    with common.timed(phase_times, 'measurement'):
      for attr in data.domain:
        clique = (attr,)
        if clique in existing_cliques:
          continue
        marginal = data.project(clique).datavector()
        noisy_marginal = (
            marginal + rng.normal(size=marginal.shape) * per_query_sigma
        )
        measurements.append(mbi.LinearMeasurement(noisy_marginal, clique))

    one_way_only = [m for m in measurements if len(m.clique) == 1]
    mappings = common.compression_mappings(
        one_way_only, self.compress_columns, constraints
    )
    if mappings:
      data = data.compress(mappings)
      measurements = [m.compress(mappings, data.domain) for m in measurements]

    logging.info(
        '[Independent]:\n%s',
        mbi.summarize(data.domain, [m.clique for m in measurements]),
    )
    with common.timed(phase_times, 'estimation'):
      estimator = mbi.estimation.MirrorDescent(self.marginal_oracle)
      model = estimator.estimate(
          data.domain,
          measurements,
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
        measurements=measurements,
        diagnostics=diagnostics,
        mappings=mappings,
    )

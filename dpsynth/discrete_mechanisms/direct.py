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

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import typing

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
import mbi
import numpy as np


@dataclasses.dataclass
class DirectMechanism(api.DPMechanism):
  """DP Mechanism that directly measures the 1-way marginals and queries."""

  marginal_oracle: mbi.MarginalOracle | None = None
  pgm_iters: int = 5000
  max_records_per_user: int = 1

  def __post_init__(self):
    api.validate_max_records_per_user(self.max_records_per_user)

  prespecified_marginal_queries: list[tuple[str, ...]] = dataclasses.field(
      default_factory=list
  )
  zcdp_rho: float | None = None

  def configure(self, *, zcdp_rho: float, **kwargs) -> DirectMechanism:
    return dataclasses.replace(self, zcdp_rho=zcdp_rho)

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    return list(self.prespecified_marginal_queries)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    if self.zcdp_rho is None:
      raise ValueError('Must call configure() before using the mechanism.')
    return dp_accounting.GaussianDpEvent(
        noise_multiplier=accounting.zcdp_gaussian_sigma(self.zcdp_rho)
    )

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      constraints: Sequence[mbi.Constraint] = (),
  ) -> common.DiscreteMechanismResult:
    if self.zcdp_rho is None:
      raise ValueError('Must call configure() before using the mechanism.')
    phase_times = {}
    selected = list(self.prespecified_marginal_queries)

    all_cliques = [m.clique for m in initial_measurements or []] + list(
        selected
    )
    logging.info(
        '[%s]:\n%s',
        type(self).__name__,
        mbi.summarize(data.domain, all_cliques),
    )

    estimator = mbi.estimation.MirrorDescent(self.marginal_oracle)
    futures = None
    try:
      futures = estimator.precompile(
          data.domain,
          list(initial_measurements or []),
          extra_cliques=list(selected),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('Precompile failed (non-fatal): %s', e)

    measurements = list(initial_measurements or [])
    if selected:
      with common.timed(phase_times, 'measurement'):
        sigma = accounting.zcdp_gaussian_sigma(self.zcdp_rho)
        measurements.extend(
            common.measure_marginals_with_noise(
                rng,
                data,  # pyrefly: ignore[bad-argument-type]
                selected,
                sigma,
                max_records_per_user=self.max_records_per_user,
            )
        )

    with common.timed(phase_times, 'estimation'):
      if futures is not None:
        try:
          futures.result()
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning('Precompile wait failed (non-fatal): %s', e)
      model = estimator.estimate(
          data.domain,
          measurements,
          iters=self.pgm_iters,
          callback_fn=mbi.callbacks.default(measurements, data.domain),
          constraints=constraints,
      )

      model = typing.cast(mbi.MarkovRandomField, model)

    diagnostics = common.clique_stats(model)
    diagnostics.phase_times = phase_times

    return common.DiscreteMechanismResult(
        model=model,
        synthetic_data=model.synthetic_data(),
        measurements=measurements,
        diagnostics=diagnostics,
    )

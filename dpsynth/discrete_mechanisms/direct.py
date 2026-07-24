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
import dataclasses

import dp_accounting
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import base
import mbi


@dataclasses.dataclass
class DirectMechanism(base.DiscreteMechanism):
  """Configuration for the direct mechanism.

  The direct mechanism measures a prespecified set of marginal queries,
  allocating the entire privacy budget to those measurements.  It does not
  measure its own one-way marginals, but can incorporate externally supplied
  ``initial_measurements`` (e.g. compressed one-ways from an orchestration
  layer) at no additional budget cost.

  Attributes:
    prespecified_marginal_queries: A list of k-way marginals that a user has
      specified.  Only these will be measured with privacy budget.
    one_way_budget_fraction: Fraction of the zCDP budget allocated to one-way
      marginals.  Overridden to 0.0 because this mechanism does not measure its
      own one-way marginals.
  """

  prespecified_marginal_queries: list[tuple[str, ...]] = dataclasses.field(
      default_factory=list
  )
  one_way_budget_fraction: float = 0.0

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the prespecified marginal queries."""
    return list(self.prespecified_marginal_queries)

  def _allocate_budget(self, remaining_rho: float) -> Mapping[str, float]:
    """Allocates the full remaining budget to the prespecified queries."""
    return {'measurement_rho': remaining_rho}

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the direct mechanism."""
    self._check_calibration()
    return dp_accounting.GaussianDpEvent(
        noise_multiplier=accounting.zcdp_gaussian_sigma(self.measurement_rho)
    )

  def _select(self, rng, data, measurements, phase_times):
    return list(self.prespecified_marginal_queries)

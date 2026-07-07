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
from dpsynth.discrete_mechanisms import base
import mbi


@dataclasses.dataclass
class IndependentMechanism(base.DiscreteMechanism):
  """Measures only one-way marginals, allocating the entire budget to them."""

  one_way_budget_fraction: float = 1.0

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the one-way marginals this mechanism will measure."""
    return [(a,) for a in domain.attributes]

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the independent mechanism."""
    self._check_calibration()
    return dp_accounting.GaussianDpEvent(
        noise_multiplier=accounting.zcdp_gaussian_sigma(self.one_way_rho)
    )

  def _select(self, rng, data, measurements, phase_times):
    return []

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

"""A generic mechanism interface and broadly useful helper functions."""

from typing import Protocol

import dp_accounting
import mbi

from . import aim
from . import aim_gdp
from . import direct
from . import independent
from . import mst
from . import swift


class DiscreteMechanismConfig(Protocol):
  """A generic mechanism configuration that operates on discrete data.

  Note: For consistency across the library, discrete mechanisms can be run
  with a given zCDP budget (rho). However, a more precise characterization
  of the privacy properties of the mechanism is given by the ``dp_event``
  method defined on this class. Given a ``zcdp_rho`` value passed into
  ``run_mechanism``, this returns a DpEvent that characterizes the privacy
  properties of the mechanism. For instance, if the mechanism satisfies mu-GDP,
  then the returned DpEvent will be
  ``dp_accounting.GaussianDpEvent(sigma=math.sqrt(0.5 / zcdp_rho))``. By using
  PLD accounting, one can use this ``dp_event`` to compute tighter
  (epsilon, delta) guarantees, or calibrate the ``zcdp_rho`` value for a
  desired (epsilon, delta) guarantee.
  """

  def dp_event(self, zcdp_rho: float) -> dp_accounting.DpEvent:
    """Returns the DP event for the mechanism."""


def run_mechanism(
    data: mbi.Projectable,
    config: DiscreteMechanismConfig,
    zcdp_rho: float,
    *,
    initial_measurements: list[mbi.LinearMeasurement] | None = None,
    initial_potentials: mbi.CliqueVector | None = None,
) -> mbi.MarkovRandomField:
  """Runs a discrete mechanism with the given configuration and privacy parameter."""
  if isinstance(config, aim.AIMConfig):
    run_mechanism_fn = aim.run_mechanism
  elif isinstance(config, aim_gdp.AIMGDPConfig):
    run_mechanism_fn = aim_gdp.run_mechanism
  elif isinstance(config, mst.MSTConfig):
    run_mechanism_fn = mst.run_mechanism
  elif isinstance(config, direct.DirectConfig):
    run_mechanism_fn = direct.run_mechanism
  elif isinstance(config, independent.IndependentConfig):
    run_mechanism_fn = independent.run_mechanism
  elif isinstance(config, independent.SWIFTConfig):
    run_mechanism_fn = swift.run_mechanism
  else:
    raise ValueError(f'Unknown mechanism: {type(config)}')

  return run_mechanism_fn(
      data,
      config,
      zcdp_rho=zcdp_rho,
      initial_measurements=initial_measurements,
      initial_potentials=initial_potentials,
  )

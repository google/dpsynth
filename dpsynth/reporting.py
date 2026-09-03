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

"""Privacy reporting utilities for DP Synth."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools

import dp_accounting

DEFAULT_TARGET_DELTAS = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


@dataclasses.dataclass(frozen=True)
class PrivacyReport:
  """Privacy guarantees computed from a differential privacy event.

  This dataclass is intended to provide a comprehensive summary of the privacy
  guarantees of a given mechanism, rather than distilling the the guarantees to
  a single (epsilon, delta) pair. This dataclass is likely to expand over time
  to include additional metrics, trade-off curves, and other information that is
  helpful to characterize the privacy properties of a mechanism.
  """

  dp_event: dp_accounting.DpEvent
  epsilon_deltas: tuple[tuple[float, float], ...]
  gdp_estimate: float | None = None
  # All mechanisms in dpsynth assume the ADD_OR_REMOVE_ONE neighboring relation.
  neighboring_relation: str = 'ADD_OR_REMOVE_ONE'

  @classmethod
  def from_dp_event(
      cls,
      event: dp_accounting.DpEvent,
      *,
      target_deltas: float | Sequence[float] = DEFAULT_TARGET_DELTAS,
      orders: Sequence[float] | None = None,
      value_discretization_interval: float | None = None,
  ) -> 'PrivacyReport':
    """Computes a PrivacyReport from a DpEvent.

    Evaluates privacy guarantees across multiple evaluation points taking the
    tightest (minimum) epsilon across supported accountants (PLD and RDP).
    Computes the Gaussian Differential Privacy (GDP) parameter estimate via PLD
    when supported.

    Args:
      event: The DpEvent to analyze.
      target_deltas: Evaluation delta(s) for (epsilon, delta)-DP.
      orders: Optional Renyi differential privacy orders for RdpAccountant.
      value_discretization_interval: Optional discretization interval for
        PLDAccountant.

    Returns:
      A PrivacyReport containing computed privacy guarantees.
    """

    if isinstance(target_deltas, (int, float)):
      target_deltas = [float(target_deltas)]

    assert all(0 < d <= 1 for d in target_deltas), 'target_deltas not in (0, 1]'

    if orders is not None:
      rdp_acc = functools.partial(
          dp_accounting.rdp.RdpAccountant, orders=orders
      )
    else:
      rdp_acc = dp_accounting.rdp.RdpAccountant

    if value_discretization_interval is not None:
      pld_acc = functools.partial(
          dp_accounting.pld.PLDAccountant,
          value_discretization_interval=value_discretization_interval,
      )
    else:
      pld_acc = dp_accounting.pld.PLDAccountant

    rdp_epsilons = _get_epsilons(event, rdp_acc, target_deltas)
    pld_epsilons = _get_epsilons(event, pld_acc, target_deltas)
    best_epsilons = [min(e) for e in zip(rdp_epsilons, pld_epsilons)]

    try:
      gdp_estimate = pld_acc().compose(event).get_gdp_parameter_estimate()
    except (dp_accounting.UnsupportedEventError, NotImplementedError):
      gdp_estimate = None


    return cls(
        dp_event=event,
        epsilon_deltas=tuple(zip(best_epsilons, target_deltas)),
        gdp_estimate=float(gdp_estimate) if gdp_estimate else None,
    )


def _get_epsilons(
    event: dp_accounting.DpEvent,
    make_accountant: Callable[[], dp_accounting.PrivacyAccountant],
    target_deltas: Sequence[float],
) -> list[float]:
  """Returns the epsilon values for the given deltas and accountant."""
  try:
    epsilons: list[float] = []
    accountant = make_accountant().compose(event)
    for delta in target_deltas:
      epsilons.append(float(accountant.get_epsilon(delta)))
    return epsilons
  except (dp_accounting.UnsupportedEventError, NotImplementedError):
    return [float('inf') for _ in target_deltas]

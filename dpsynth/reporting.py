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

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
from typing import Any

import dp_accounting
import numpy as np

DEFAULT_TARGET_DELTAS = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
DEFAULT_TARGET_FALSE_POSITIVE_RATES = (
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    0.05,
    0.1,
    0.2,
    0.5,
)


@dataclasses.dataclass(frozen=True)
class PrivacyReport:
  """Privacy guarantees computed from a differential privacy event.

  This dataclass is intended to provide a comprehensive summary of the privacy
  guarantees of a given mechanism, rather than distilling the guarantees to a
  single (epsilon, delta) pair. This dataclass is likely to expand over time to
  include additional metrics, trade-off curves, and other information that is
  helpful to characterize the privacy properties of a mechanism.
  """

  dp_event: dp_accounting.DpEvent
  epsilon_deltas: tuple[tuple[float, float], ...]
  gdp_estimate: float | None = None
  trade_off_curve: tuple[tuple[float, float], ...] | None = None
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
  # All mechanisms in dpsynth assume the ADD_OR_REMOVE_ONE neighboring relation.
  neighboring_relation: str = 'ADD_OR_REMOVE_ONE'

  @property
  def non_dp_disclosures(self) -> tuple[str, ...]:
    """Returns non-DP disclosures tracked in metadata, if any."""
    disclosures = self.metadata.get('non_dp_disclosures', ())
    return tuple(disclosures)

  @classmethod
  def from_dp_event(
      cls,
      event: dp_accounting.DpEvent,
      *,
      target_deltas: float | Sequence[float] = DEFAULT_TARGET_DELTAS,
      target_false_positive_rates: float | Sequence[float] | None = (
          DEFAULT_TARGET_FALSE_POSITIVE_RATES
      ),
      orders: Sequence[float] | None = None,
      value_discretization_interval: float | None = None,
      metadata: Mapping[str, Any] | None = None,
      non_dp_disclosures: Sequence[str] | None = None,
  ) -> 'PrivacyReport':
    """Computes a PrivacyReport from a DpEvent.

    Evaluates privacy guarantees across multiple evaluation points taking the
    tightest (minimum) epsilon across supported accountants (PLD and RDP).
    Computes the Gaussian Differential Privacy (GDP) parameter estimate and
    hypothesis testing trade-off curve (TPR vs. FPR) via PLD when supported.

    Args:
      event: The DpEvent to analyze.
      target_deltas: Evaluation delta(s) for (epsilon, delta)-DP.
      target_false_positive_rates: Optional evaluation FPR(s) for hypothesis
        testing trade-off curve (TPR vs. FPR).
      orders: Optional Renyi differential privacy orders for RdpAccountant.
      value_discretization_interval: Optional discretization interval for
        PLDAccountant.
      metadata: Optional metadata dictionary capturing custom notes or context
        about the privacy guarantee.
      non_dp_disclosures: Optional sequence of non-DP disclosures or custom
        notes to store in metadata.

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
      pld_accountant = pld_acc().compose(event)
      gdp_estimate = pld_accountant.get_gdp_parameter_estimate()
    except (dp_accounting.UnsupportedEventError, NotImplementedError):
      pld_accountant = None
      gdp_estimate = None

    trade_off_curve = None
    if pld_accountant is not None and target_false_positive_rates is not None:
      if isinstance(target_false_positive_rates, (int, float)):
        target_fprs = [float(target_false_positive_rates)]
      else:
        target_fprs = [float(x) for x in target_false_positive_rates]
      assert all(
          0 <= f <= 1 for f in target_fprs
      ), 'target_false_positive_rates not in [0, 1]'
      try:
        tprs = pld_accountant.get_true_positive_rates(np.asarray(target_fprs))
        tpr_list = (
            [float(tprs)]
            if isinstance(tprs, (int, float))
            else [float(t) for t in tprs]
        )
        trade_off_curve = tuple(zip(tpr_list, target_fprs))
      except (dp_accounting.UnsupportedEventError, NotImplementedError):
        trade_off_curve = None

    report_metadata = dict(metadata or {})
    if non_dp_disclosures is not None:
      report_metadata['non_dp_disclosures'] = list(non_dp_disclosures)

    return cls(
        dp_event=event,
        epsilon_deltas=tuple(zip(best_epsilons, target_deltas)),
        gdp_estimate=float(gdp_estimate) if gdp_estimate is not None else None,
        trade_off_curve=trade_off_curve,
        metadata=report_metadata,
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

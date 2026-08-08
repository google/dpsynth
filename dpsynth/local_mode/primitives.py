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

"""Differentially private primitives for quantiles and partition selection.

These implementations only depend on numpy and scipy and utilize vectorized
operations for efficiency in single-machine environments.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import dp_accounting
from dpsynth import api
from dpsynth.local_mode import _quantiles
import numpy as np
import scipy.stats

CalibratedMechanism = api.CalibratedMechanism
MechanismConfig = api.MechanismConfig
DPMechanism = api.DPMechanism


@dataclasses.dataclass
class HistogramResult:
  """Result of a differentially private histogram computation."""

  counts: np.ndarray


@dataclasses.dataclass
class PartitionSelectionResult:
  """Result of differentially private partition selection."""

  selected_partitions: np.ndarray
  estimated_counts: np.ndarray


def _contribution_bound(prng, user_ids, max_part):
  """Return array idx where all ids appear <=max_part times in user_ids[idx]."""
  # Sort by ID + noise to shuffle within groups. Then find where
  # groups start/end, and select the first max_part elements of each group.
  # Use lexsort with random keys to shuffle string/object IDs safely.
  random_keys = prng.uniform(size=user_ids.size)
  idx = np.lexsort((random_keys, user_ids))
  sorted_ids = user_ids[idx]
  diff = np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]
  kernel = np.ones(max_part, dtype=bool)
  # This convolution determines if any of previous max_part elements are True.
  mask = np.convolve(diff, kernel, mode='full')[: user_ids.size]
  return idx[mask]


def _get_threshold(delta, sigma, max_part):
  ks = np.arange(1, max_part + 1)
  failure_prob = (1 - delta) ** (1 / ks)
  thresholds = 1 / np.sqrt(ks) + sigma * scipy.stats.norm.ppf(failure_prob)
  return thresholds.max()


def select_partitions_gaussian_thresholding(
    rng: np.random.Generator,
    data: np.ndarray,
    gdp_budget: float,
    delta: float,
    min_count: int = 1,
    max_records_per_user: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Selects partitions using Gaussian Thresholding (Weighted Gaussian).

  This implements Algorithm 2 from the DP-SIPS paper (Swanberg et al., 2023)
  under item-level DP. It is the simplest partition selection mechanism:

    1. Compute the histogram of partition counts.
    2. Add Gaussian noise calibrated to the privacy budget.
    3. Return partitions whose noisy count exceeds a threshold chosen to
       bound the false-positive probability per empty partition at delta.

  Under item-level DP each record is treated as a distinct user contributing
  to exactly one partition, so the histogram has L2 sensitivity 1.  The
  threshold is T = min_count + sigma * Phi^{-1}(1 - delta), following the
  paper's formula with max_part = 1 and a shift of (min_count - 1) to
  account for the minimum count guarantee.

  When ``min_count > 1``, partitions with true count below ``min_count``
  are pre-filtered and the threshold shifts up accordingly. The privacy
  guarantee is preserved: partitions where both neighboring datasets are
  above ``min_count`` are covered by the Gaussian mechanism, and the
  boundary case (one dataset at ``min_count - 1``, the other at
  ``min_count``) is covered by the same additive delta.

  When ``max_records_per_user > 1`` the mechanism switches to user-level DP
  via a naive, conservative reduction: a single user may place all ``k``
  records in one partition, so the histogram's L2 sensitivity grows to ``k``.
  Both the noise standard deviation and the threshold are scaled by ``k``,
  which is equivalent to running the item-level mechanism with ``k`` times the
  sigma and threshold. This is sound but suboptimal -- a user->record mapping
  would allow tighter per-user contribution bounding (e.g. capping the number
  of distinct partitions a user touches) and hence far better utility.

  Args:
    rng: A numpy random number generator.
    data: 1D array of integers, where each element is a partition ID.
    gdp_budget: Privacy budget in terms of squared Gaussian DP mu parameter
      (gdp_budget = mu^2 = 1 / sigma^2).
    delta: Failure probability (false positive bound per empty partition).
    min_count: Minimum true count for a partition to be eligible. Partitions
      with fewer occurrences in the data are never returned. Must be >= 1.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Added noise (and mechanism sensitivity) is scaled by
      this factor to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.

  Returns:
    A tuple containing:
      - selected_partitions: 1D array of partition IDs that passed the
        threshold.
      - estimated_counts: 1D array of noisy counts for each selected
        partition.
      - stddev: The standard deviation of the Gaussian noise added
        (``max_records_per_user * sigma``).
  """
  if gdp_budget <= 0 or delta <= 0:
    raise ValueError(f'{gdp_budget=} and {delta=} must be positive.')
  if min_count < 1:
    raise ValueError(f'{min_count=} must be >= 1.')

  stddev = max_records_per_user / np.sqrt(gdp_budget)

  if data.size == 0:
    return np.empty(0, dtype=data.dtype), np.empty(0, dtype=float), stddev

  unique_parts, counts = np.unique(data, return_counts=True)

  # Filter partitions below the minimum count before adding noise.
  above_min = counts >= min_count
  unique_parts, counts = unique_parts[above_min], counts[above_min]
  if unique_parts.size == 0:
    return np.empty(0, dtype=data.dtype), np.empty(0, dtype=float), stddev

  noisy_counts = counts + rng.normal(scale=stddev, size=counts.size)

  # A partition that is a candidate here but absent from a neighbor drives the
  # per-partition false-positive budget `delta`. One user contributes up to
  # k = max_records_per_user records, so (i) the noise std is
  # stddev = k / sqrt(gdp_budget), and (ii) such a partition's true count can
  # reach (min_count - 1) + k -- the neighbor sits just under the eligibility
  # cutoff at min_count - 1 and the user piles all k records into it. Bounding
  #   Pr[(min_count - 1 + k) + N(0, stddev^2) >= T] <= delta
  # gives T = (min_count + k - 1) + stddev * ppf(1 - delta).
  base = float(max_records_per_user + min_count - 1)
  threshold = base + stddev * scipy.stats.norm.ppf(1.0 - delta)
  passed = noisy_counts >= threshold
  return unique_parts[passed], noisy_counts[passed], stddev


def _select_partitions_sips(
    rng: np.random.Generator,
    data: np.ndarray,
    gdp_budget: float,
    delta: float,
    num_rounds: int | None = None,
    user_ids: np.ndarray | None = None,
    max_part: int = 1,
    allocation_factor: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Implements the DP-SIPS mechanism for partition selection.

  Args:
    rng: A numpy random number generator.
    data: 1D array of integers, where each element is a partition ID.
    gdp_budget: Total privacy budget in terms of squared Gaussian DP mu
      parameter (gdp_budget = mu^2 = 1 / sigma^2).
    delta: Failure probability (false positive bound per empty partition).
    num_rounds: Number of rounds to run the mechanism. Defaults to 1 if user_ids
      is None, and 3 otherwise.
    user_ids: Optional 1D array of user IDs corresponding to data. If provided,
      user-level DP is guaranteed. If None, item-level DP is guaranteed
      (assuming each record is a unique user).
    max_part: Maximum number of partitions any single user can contribute to in
      a single round.
    allocation_factor: Factor by which to increase the budget each round.

  Returns:
    A tuple containing:
      - selected_partitions: 1D array of unique partition IDs that passed the
        threshold.
      - estimated_counts: 1D array of noisy (or weighted noisy) counts for each
        selected partition in the round it was discovered.
      - standard_deviation: A single float representing the uniform standard
        deviation of the noise added to the estimated counts.
  """
  if num_rounds is None:
    num_rounds = 1 if user_ids is None else 3
  if num_rounds <= 0:
    raise ValueError(f'num_rounds ({num_rounds}) must be greater than 0.')
  if gdp_budget <= 0 or delta <= 0:
    raise ValueError(f'{gdp_budget=} and {delta=} must be positive.')

  fractions = allocation_factor ** np.arange(num_rounds)[::-1]
  fractions /= fractions.sum()
  gdp_rounds, delta_rounds = gdp_budget * fractions, delta * fractions
  sigma_rounds = 1.0 / np.sqrt(gdp_rounds)
  max_sigma = float(np.max(sigma_rounds))

  if data.size == 0:
    return np.empty(0, dtype=data.dtype), np.empty(0, dtype=float), max_sigma

  if user_ids is None:
    user_ids = np.arange(data.size)
  if user_ids.size != data.size:
    raise ValueError('user_ids must have the same size as data.')

  combined = np.stack((user_ids, data), axis=1)
  unique_combined = np.unique(combined, axis=0)
  rem_user_ids = unique_combined[:, 0]
  rem_partitions = unique_combined[:, 1]

  selected_partitions = []
  selected_counts = []
  for i in range(num_rounds):
    if rem_partitions.size == 0:
      break

    threshold = _get_threshold(delta_rounds[i], sigma_rounds[i], max_part)

    mask = _contribution_bound(rng, rem_user_ids, max_part)
    curr_user_ids = rem_user_ids[mask]
    curr_partitions = rem_partitions[mask]

    unique_users, user_counts = np.unique(curr_user_ids, return_counts=True)
    user_to_count = dict(zip(unique_users, user_counts))
    weights = np.array([1.0 / user_to_count[u] ** 0.5 for u in curr_user_ids])

    unique_parts, inverse_indices = np.unique(
        curr_partitions, return_inverse=True
    )
    weighted_counts = np.bincount(inverse_indices, weights=weights)
    noised_counts = rng.normal(weighted_counts, scale=sigma_rounds[i])

    passed_mask = noised_counts >= threshold
    round_selections = unique_parts[passed_mask]
    round_counts = noised_counts[passed_mask]
    if round_selections.size > 0:
      selected_partitions.append(round_selections)
      selected_counts.append(round_counts)

      mask = ~np.isin(rem_partitions, round_selections)
      rem_user_ids = rem_user_ids[mask]
      rem_partitions = rem_partitions[mask]

  if not selected_partitions:
    return (
        np.empty(0, dtype=data.dtype),
        np.empty(0, dtype=float),
        max_sigma,
    )
  selected_partitions = np.concatenate(selected_partitions)
  selected_counts = np.concatenate(selected_counts)
  return selected_partitions, selected_counts, max_sigma


@dataclasses.dataclass(frozen=True)
class DPQuantilesConfig(MechanismConfig):
  """Recipe for differentially private quantiles.

  Attributes:
    num_partitions: Number of quantile partitions (must be a power of 2).
    lower: Lower bound of the data domain.
    upper: Upper bound of the data domain (exclusive).
    jitter_strategy: Tie-breaking jitter passed to ``quantiles_from_histogram``:
      ``'refine'`` for integer attributes, ``'symmetric'`` for continuous ones.
    epsilon_ratio: Factor by which epsilon grows at each deeper level.
  """

  num_partitions: int
  lower: float
  upper: float
  jitter_strategy: Literal['symmetric', 'refine'] = 'symmetric'
  epsilon_ratio: float = 2.0

  @property
  def _num_levels(self) -> int:
    result = int(np.log2(self.num_partitions))
    if 2**result != self.num_partitions:
      raise ValueError(f'{self.num_partitions=} must be a power of 2.')
    return result

  def configure(self, *, zcdp_rho, delta=0.0, max_records_per_user=1):  # pyrefly: ignore[bad-override]
    """Returns a calibrated mechanism for the given zCDP budget."""
    levels = self._num_levels
    rho_ratio = self.epsilon_ratio**2
    budget_weights = rho_ratio ** np.arange(levels)[::-1]
    rho_levels = zcdp_rho * budget_weights / budget_weights.sum()
    eps = np.sqrt(8.0 * rho_levels)
    return DPQuantiles(self, tuple(eps.tolist()), max_records_per_user)


@dataclasses.dataclass(frozen=True)
class DPQuantiles(CalibratedMechanism):
  """Calibrated DP quantiles via composed exponential mechanisms.

  Computes quantile edges by recursive median bisection on a dense histogram.
  The ``__call__`` method takes a 1D histogram of counts and returns the
  quantile edge values.

  Attributes:
    config: The recipe this mechanism was calibrated from.
    epsilon_levels: Per-level exponential-mechanism epsilons.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. The per-level epsilons are divided by this factor at run
      time to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.
  """

  config: DPQuantilesConfig
  epsilon_levels: tuple[float, ...]
  max_records_per_user: int = 1

  def __post_init__(self):
    api.validate_max_records_per_user(self.max_records_per_user)

  @property
  def zcdp_rho(self) -> float:
    """Total zCDP rho consumed, derived from the per-level epsilons."""
    return sum(e**2 / 8.0 for e in self.epsilon_levels)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for the quantile computation."""
    return dp_accounting.ComposedDpEvent([
        dp_accounting.ExponentialMechanismDpEvent(epsilon=float(eps))
        for eps in self.epsilon_levels
    ])

  def __call__(
      self, rng: np.random.Generator, counts: np.ndarray
  ) -> list[float]:
    """Returns quantile edges from a dense histogram of counts."""
    eps_levels = np.asarray(self.epsilon_levels) / self.max_records_per_user
    indices = _quantiles.quantiles_from_histogram(
        rng, counts, eps_levels, self.config.jitter_strategy
    )
    # Map cell indices back to domain values; delta is the grid step, which
    # equals the integer step for integer attributes so edges stay integer.
    delta = (self.config.upper - self.config.lower) / max(1, counts.size - 1)
    return [self.config.lower + i * delta for i in indices]


@dataclasses.dataclass(frozen=True)
class DPGaussianHistogramConfig(MechanismConfig):
  """Recipe for a differentially private histogram via the Gaussian mechanism.

  Attributes:
    domain_size: Number of categories in the histogram domain.
  """

  domain_size: int

  def configure(self, *, zcdp_rho, delta=0.0, max_records_per_user=1):  # pyrefly: ignore[bad-override]
    """Returns a calibrated mechanism with sigma derived from the budget."""
    sigma = math.sqrt(0.5 / zcdp_rho)
    return DPGaussianHistogram(self, sigma, max_records_per_user)


@dataclasses.dataclass(frozen=True)
class DPGaussianHistogram(CalibratedMechanism):
  """Calibrated DP histogram via the Gaussian mechanism.

  The natural privacy parameter is ``sigma``, the noise standard deviation.
  The conversion from zCDP is ``sigma = sqrt(0.5 / zcdp_rho)``.

  Attributes:
    config: The recipe this mechanism was calibrated from.
    sigma: Gaussian noise standard deviation.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Added noise (and mechanism sensitivity) is scaled by
      this factor to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.
  """

  config: DPGaussianHistogramConfig
  sigma: float
  max_records_per_user: int = 1

  def __post_init__(self):
    api.validate_max_records_per_user(self.max_records_per_user)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)

  def __call__(
      self, rng: np.random.Generator, counts: np.ndarray
  ) -> HistogramResult:
    """Adds Gaussian noise to the given counts."""
    noise = rng.normal(
        scale=self.max_records_per_user * self.sigma,
        size=self.config.domain_size,
    )
    return HistogramResult(counts=counts.astype(float) + noise)


@dataclasses.dataclass(frozen=True)
class DPGaussianCountConfig(MechanismConfig):
  """Recipe for a differentially private count via the Gaussian mechanism."""

  def configure(self, *, zcdp_rho, delta=0.0, max_records_per_user=1):  # pyrefly: ignore[bad-override]
    """Returns a calibrated mechanism with sigma derived from the budget."""
    sigma = math.sqrt(0.5 / zcdp_rho)
    return DPGaussianCount(sigma, max_records_per_user)


@dataclasses.dataclass(frozen=True)
class DPGaussianCount(CalibratedMechanism):
  """Calibrated DP count via the Gaussian mechanism.

  Attributes:
    sigma: Gaussian noise standard deviation.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes; added noise is scaled by this factor for user-level DP.
  """

  sigma: float
  max_records_per_user: int = 1

  def __post_init__(self):
    api.validate_max_records_per_user(self.max_records_per_user)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)

  def noisy_count(self, rng: np.random.Generator, true_count: int) -> float:
    """Returns ``true_count`` plus calibrated Gaussian noise."""
    return float(
        true_count + rng.normal(scale=self.max_records_per_user * self.sigma)
    )

  def __call__(self, rng: np.random.Generator, data: np.ndarray) -> float:
    """Returns a noisy count of len(data) + Gaussian noise."""
    # Delegate to noisy_count so there is a single noise implementation.
    return self.noisy_count(rng, len(data))


@dataclasses.dataclass(frozen=True)
class DPPartitionSelectionConfig(MechanismConfig):
  """Recipe for differentially private partition selection.

  Attributes:
    delta: Failure probability for the thresholding step.
    min_count: Minimum true count for a partition to be returned.
  """

  delta: float
  min_count: int = 1

  def configure(self, *, zcdp_rho, delta=0.0, max_records_per_user=1):  # pyrefly: ignore[bad-override]
    """Returns a calibrated mechanism with sigma derived from the budget."""
    sigma = math.sqrt(0.5 / zcdp_rho)
    return DPPartitionSelection(self, sigma, max_records_per_user)


@dataclasses.dataclass(frozen=True)
class DPPartitionSelection(CalibratedMechanism):
  """Calibrated DP partition selection via Gaussian Thresholding.

  Because partition selection is an approximate (delta > 0) mechanism, the
  ``dp_event`` composes the Gaussian event with an ``EpsilonDeltaDpEvent``
  representing the additive thresholding delta.

  When ``max_records_per_user > 1`` the mechanism uses a naive, conservative
  user-level reduction (noise and threshold scaled by that factor): sound but
  suboptimal, since tighter bounding would require a user->record mapping.

  Attributes:
    config: The recipe this mechanism was calibrated from.
    sigma: Gaussian noise standard deviation.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Added noise (and mechanism sensitivity) is scaled by
      this factor to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.
  """

  config: DPPartitionSelectionConfig
  sigma: float
  max_records_per_user: int = 1

  def __post_init__(self):
    api.validate_max_records_per_user(self.max_records_per_user)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the privacy event including thresholding delta."""
    main_event = dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)
    failure_event = dp_accounting.dp_event.EpsilonDeltaDpEvent(
        0, self.config.delta
    )
    return dp_accounting.ComposedDpEvent([main_event, failure_event])

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> PartitionSelectionResult:
    """Runs partition selection on integer-encoded partition IDs."""
    gdp_budget = np.inf if self.sigma == 0.0 else 1.0 / (self.sigma**2)
    parts, counts, _ = select_partitions_gaussian_thresholding(
        rng,
        data,
        gdp_budget,
        self.config.delta,
        min_count=self.config.min_count,
        max_records_per_user=self.max_records_per_user,
    )
    return PartitionSelectionResult(parts, counts)

  def from_summary(
      self,
      rng: np.random.Generator,
      counts: np.ndarray,
  ) -> PartitionSelectionResult:
    """Single-round partition selection from pre-aggregated counts.

    Args:
      rng: A numpy random number generator.
      counts: 1D array of per-partition counts.

    Returns:
      A PartitionSelectionResult with indices into `counts` as the
      selected_partitions and their noisy counts.
    """
    above_min = counts >= self.config.min_count
    eligible_idx = np.where(above_min)[0]
    eligible_counts = counts[above_min].astype(float)
    stddev = self.max_records_per_user * self.sigma
    noisy_counts = eligible_counts + rng.normal(
        scale=stddev, size=len(eligible_counts)
    )
    # select_partitions_gaussian_thresholding; the two must stay in sync.
    # Tight shift: one user can push a partition's count to
    # (min_count - 1) + max_records_per_user (see that function for the
    # full derivation).
    base = float(self.max_records_per_user + self.config.min_count - 1)
    threshold = base + stddev * scipy.stats.norm.ppf(1.0 - self.config.delta)
    passed = noisy_counts >= threshold
    return PartitionSelectionResult(eligible_idx[passed], noisy_counts[passed])

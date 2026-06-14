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

These implementations only depened on numpy and scipy and utilize vectorized
operations for efficiency in single-machine environments.
"""

import numpy as np
import scipy.special
import scipy.stats


def median(
    rng: np.random.Generator,
    data: np.ndarray,
    lower: float,
    upper: float,
    zcdp_rho: float,
    jitter_multiple: float = 1e-4,
    num_examples_per_user: int = 1,
) -> float:
  """Computes a differentially private median using the exponential mechanism.

  This function implements the continuous rank-based DP median. Candidates are
  the intervals between sorted data points. The utility of an interval is based
  on the distance of its rank from N/2.

  This mechanism is an instance of the exponential mechanism with parameter
  epsilon = sqrt(8 * zcdp_rho) and sensitivity = num_examples_per_user.

  Args:
    rng: A numpy random number generator.
    data: 1D array of numerical data.
    lower: Lower bound for the data.
    upper: Upper bound for the data.
    zcdp_rho: Total zCDP privacy budget for the median call.
    jitter_multiple: Multiplier for the jitter scale, relative to upper-lower.
    num_examples_per_user: Number of examples per user. If provided, this
      mechanism satisfies user-level DP.

  Returns:
    A differentially private median estimate.
  """
  if lower > upper:
    raise ValueError(f"{lower=} cannot be greater than {upper=}.")

  clamped_data = np.clip(data, lower, upper)
  n = clamped_data.size

  if zcdp_rho == np.inf:
    if n == 0:
      return (lower + upper) / 2
    return float(np.median(clamped_data))

  # Jitter size proportional to range. A small jitter makes duplicates unique
  # and gives them non-zero length intervals, allowing them to be sampled.
  jitter_scale = (upper - lower) * jitter_multiple
  jitter = rng.uniform(-jitter_scale, jitter_scale, size=clamped_data.size)
  jittered_data = np.clip(clamped_data + jitter, lower, upper)

  sorted_data = np.sort(jittered_data)
  n = sorted_data.size
  x = np.r_[lower, sorted_data, upper]
  lengths = np.diff(x)
  ranks = np.arange(n + 1)
  utilities = -np.abs(ranks - n / 2)

  # Convert zCDP rho to exponential mechanism parameter.
  epsilon = np.sqrt(8 * zcdp_rho)
  sensitivity = num_examples_per_user
  alpha = epsilon / sensitivity

  # Compute output probabilities for each interval.
  probs = scipy.special.softmax(np.log(lengths) + alpha * utilities)

  # Sample an interval index, and a value uniformly from the interval.
  interval_idx = rng.choice(n + 1, p=probs)
  v_min = x[interval_idx]
  v_max = x[interval_idx + 1]
  return rng.uniform(v_min, v_max)


def quantiles(
    rng: np.random.Generator,
    data: np.ndarray,
    lower: float,
    upper: float,
    num_partitions: int,
    zcdp_rho: float,
    num_examples_per_user: int = 1,
) -> list[float]:
  """Computes uniformly spaced differentially private quantiles.

  This function is a log2(num_partitions) composition of the exponential
  mechanism where the fraction of the total zCDP budget assigned to each level
  is proportional to 0.25^level.

  Args:
    rng: A numpy random number generator.
    data: 1D array of numerical data.
    lower: Lower bound for the data.
    upper: Upper bound for the data.
    num_partitions: Number of partitions (n) to compute (must be a power of 2).
      This function computes n-1 quantiles for [k, 2*k, ..., (n-1)*k] where  k =
      1/n, corresponding to the set of n intervals [lower, k), [k, 2k), ...,
      [k*(n-1), upper).
    zcdp_rho: Total zCDP privacy budget for the quantiles call.
    num_examples_per_user: Number of examples per user. If provided, this
      mechanism satisfies user-level DP.

  Returns:
    A length (num_partitions-1) sorted list of private quantile estimates.
  """
  if num_partitions <= 0 or (num_partitions & (num_partitions - 1)) != 0:
    raise ValueError(f"num_buckets ({num_partitions}) must be a power of 2.")

  if num_examples_per_user != 1:
    # It is not obvious if the parallel composition logic holds below when users
    # may contribute a subset of their data to multiple partitions.
    raise ValueError(f"{num_examples_per_user=} is not currently supported.")

  levels = int(np.log2(num_partitions))
  if levels == 0:
    return []

  # Split the budget so that each level gets noise proportional to data size.
  # rho_1 + ... + rho_levels = rho
  # rho_i = 4 * rho_{i+1}

  budget_weights = 4 ** np.arange(levels)[::-1]
  rho_levels = zcdp_rho * budget_weights / budget_weights.sum()

  def quantiles_rec(current_data, curr_lower, curr_upper, current_depth):
    if current_depth == 0:
      return []

    rho_level = rho_levels[current_depth - 1]
    med = median(rng, current_data, curr_lower, curr_upper, rho_level)

    left_mask = current_data <= med
    left_data = current_data[left_mask]
    right_data = current_data[~left_mask]

    left_points = quantiles_rec(left_data, curr_lower, med, current_depth - 1)
    right_points = quantiles_rec(right_data, med, curr_upper, current_depth - 1)

    return left_points + [med] + right_points

  return quantiles_rec(data, lower, upper, levels)


def _contribution_bound(prng, user_ids, max_part):
  """Return array idx where all ids appear <=max_part times in user_ids[idx]."""
  # Sort by ID + noise to shuffle within groups. Then find where
  # groups start/end, and select the first max_part elements of each group.
  idx = np.argsort(user_ids + prng.uniform(size=user_ids.size))
  sorted_ids = user_ids[idx]
  diff = np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]
  kernel = np.ones(max_part, dtype=bool)
  # This convolution determines if any of previous max_part elements are True.
  mask = np.convolve(diff, kernel, mode="full")[: user_ids.size]
  return idx[mask]


def _get_threshold(delta, sigma, max_part):
  ks = np.arange(1, max_part + 1)
  failure_prob = (1 - delta) ** (1 / ks)
  thresholds = 1 / np.sqrt(ks) + sigma * scipy.stats.norm.ppf(failure_prob)
  return thresholds.max()


def select_partitions_sips(
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
    raise ValueError(f"num_rounds ({num_rounds}) must be greater than 0.")
  if gdp_budget <= 0 or delta <= 0:
    raise ValueError(f"{gdp_budget=} and {delta=} must be positive.")

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
    raise ValueError("user_ids must have the same size as data.")

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


def gaussian_histogram(
    rng: np.random.Generator,
    data: np.ndarray,
    domain_size: int,
    sigma: float,
) -> np.ndarray:
  """Computes a noisy histogram over a closed domain using the Gaussian mechanism.

  The histogram query has L2 sensitivity 1 under item-level DP (each record
  contributes +1 to exactly one bin). Gaussian noise with the given standard
  deviation is added independently to each bin count.

  Args:
    rng: A numpy random number generator.
    data: 1D array of integer-encoded categorical values in [0, domain_size).
    domain_size: Number of categories in the closed domain.
    sigma: Standard deviation of the Gaussian noise added to each bin.

  Returns:
    A length-`domain_size` array of noisy counts.
  """
  return np.bincount(data, minlength=domain_size) + rng.normal(
      scale=sigma, size=domain_size
  )

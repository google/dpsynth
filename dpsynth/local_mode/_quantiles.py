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

"""DP quantiles from dense histograms via recursive median bisection.

This module computes differentially private quantile edges from a dense
histogram of counts, using the discrete exponential mechanism.  It works purely
in index space -- ``quantiles_from_histogram`` returns cell indices into the
histogram, and the caller maps those indices to domain values.  The primary use
case is a two-pass pipeline: a first pass computes a dense histogram over a
fine-grained grid, then ``quantiles_from_histogram`` finds DP quantile indices
from that histogram without touching individual records.

Tie handling via jitter
-----------------------
Recursive median bisection needs each record assigned to one side of every
split independently.  A "spike" of records tied on one grid cell breaks this: a
whole-cell split sends all that mass to one side, biasing the quantiles and
collapsing sub-ranges (dropping edges).  We fix this by breaking ties directly
in the histogram domain rather than over the raw data values -- each cell's
count is redistributed to nearby cells as ``Multinomial(count, kernel)`` (one
draw per non-empty cell), which is distributionally identical to independently
perturbing each record and so needs no extra privacy budget.  The ``refine``
strategy uses a strictly-positive kernel over refined sub-cells (value-
preserving); the ``symmetric`` strategy uses a symmetric kernel over neighboring
grid cells.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.special


def _median_from_histogram(
    rng: np.random.Generator,
    counts: np.ndarray,
    epsilon: float,
) -> int:
  """Returns the index of a DP median within a dense histogram.

  Args:
    rng: A numpy random number generator.
    counts: Dense 1D histogram counts.
    epsilon: Exponential mechanism privacy parameter for this level.

  Returns:
    The index of the selected median grid point within ``counts``.
  """
  total_points = len(counts)
  if total_points == 0:
    return 0
  n = counts.sum()
  target = n / 2.0
  cumsum = np.cumsum(counts)

  # Infinite budget = exact median, useful for testing.
  if epsilon == np.inf:
    return int(np.searchsorted(cumsum, target))

  # Score u(v) = -dist(target, [L_v, R_v]), sensitivity 1/2.
  left_ranks = np.r_[0, cumsum[:-1]]
  scores = -np.maximum(0, np.maximum(left_ranks - target, target - cumsum))

  probs = scipy.special.softmax(epsilon * scores)
  return int(rng.choice(total_points, p=probs))


def jitter_factor(num_partitions):
  """Returns a data-independent jitter resolution m from num_partitions."""
  # m >= num_partitions keeps each jittered cell below one partition's mass;
  # the 4x absorbs multinomial fluctuation.
  return max(1, 4 * num_partitions)


def quantiles_from_histogram(
    rng: np.random.Generator,
    counts: np.ndarray,
    epsilon_levels: np.ndarray,
    jitter_strategy: Literal['symmetric', 'refine'],
) -> list[int]:
  """DP quantile edge indices into ``counts`` via jittered median bisection.

  Operates purely in index space: it returns cell indices into ``counts`` and
  leaves the mapping from index to domain value to the caller.

  Args:
    rng: A numpy random number generator.
    counts: Dense 1D histogram counts.
    epsilon_levels: Per-level exponential mechanism epsilons, ordered from the
      deepest (finest) level to the shallowest (coarsest).
    jitter_strategy: Specifies the pre-processing jitter strategy, -
      'symmetric': jitter mass to +/- m//2 neighbors on the same grid. -
      'refine': jitter mass to m equivalent sub-cells.

  Returns:
    A sorted list of ``2 ** len(epsilon_levels) - 1`` cell indices.
  """
  counts = np.asarray(counts)
  m = jitter_factor(2 ** len(epsilon_levels))

  if jitter_strategy == 'refine':
    stride, offsets = m, np.arange(m)
  else:
    half = m // 2
    stride, offsets = 1, np.arange(-half, half + 1)

  # Scatter each cell's mass over its jittered targets: same law as perturbing
  # each record, so it breaks ties without spending extra privacy budget.
  num_cells = counts.size * stride
  nz = np.flatnonzero(counts)
  probas = np.full(offsets.size, 1.0 / offsets.size)
  split = rng.multinomial(counts[nz].astype(np.int64), probas)
  targets = np.clip(nz[:, None] * stride + offsets, 0, num_cells - 1)
  jittered = np.bincount(
      targets.flatten(), weights=split.flatten(), minlength=num_cells
  )

  def _rec(lo_idx, hi_idx, depth):
    if depth == 0:
      return []
    median_idx = lo_idx + _median_from_histogram(
        rng, jittered[lo_idx:hi_idx], epsilon_levels[depth - 1]
    )
    left = _rec(lo_idx, median_idx, depth - 1)
    right = _rec(median_idx, hi_idx, depth - 1)
    return left + [median_idx] + right

  result = _rec(0, jittered.size, len(epsilon_levels))
  if jitter_strategy == 'refine':
    result = [idx // m for idx in result]
  return result

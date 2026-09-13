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

"""CDP <> ADP conversion from "The Discrete Gaussian for Differential Privacy".

See Section 2.3 of https://arxiv.org/abs/2004.00010. This code was adapted
from https://github.com/IBM/discrete-gaussian-differential-privacy/.

Note: this file is subject to be deprecated in the near future, in favor of
using the dp_accounting library.
"""

import math
import scipy.special
import scipy.stats


def zcdp_delta(rho: float, eps: float) -> float:
  """Return the minimum value of delta such that rho-zCDP implies (epsilon, delta)-DP."""
  assert rho >= 0
  assert eps >= 0
  if rho == 0:
    return 0

  amin = 1.01
  amax = (eps + 1) / (2 * rho) + 2
  alpha = math.nan
  while amax - amin > 1e-10:
    alpha = (amin + amax) / 2
    derivative = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
    if derivative < 0:
      amin = alpha
    else:
      amax = alpha
  delta = math.exp(
      (alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)
  ) / (alpha - 1.0)
  return min(delta, 1.0)


def zcdp_eps(rho: float, delta: float) -> float:
  """Return the minimum value of epsilon such that rho-zCDP implies (epsilon, delta)-DP."""
  assert rho >= 0
  assert delta > 0
  if delta >= 1 or rho == 0:
    return 0.0
  epsmin = 0.0
  epsmax = rho + 2 * math.sqrt(rho * math.log(1 / delta))
  while epsmax - epsmin > 1e-10:
    eps = (epsmin + epsmax) / 2
    if zcdp_delta(rho, eps) <= delta:
      epsmax = eps
    else:
      epsmin = eps
  return epsmax


def zcdp_rho(eps: float, delta: float) -> float:
  """Return the maximum value of rho such that rho-zCDP implies (epsilon, delta)-DP."""
  assert eps >= 0
  assert delta > 0
  if delta >= 1:
    return 0.0
  rhomin = 0.0
  rhomax = eps + 1
  while rhomax - rhomin > 1e-10:
    rho = (rhomin + rhomax) / 2
    if zcdp_delta(rho, eps) <= delta:
      rhomin = rho
    else:
      rhomax = rho
  return rhomin


def zcdp_gaussian_sigma(rho: float) -> float:
  """Minimum sigma such that the Gaussian mechanism satisfies rho-zCDP."""
  # rho = 0.5 / sigma^2
  return math.sqrt(0.5 / rho)


def zcdp_exponential_eps(rho: float) -> float:
  """Maximum epsilon such that the exponential mechanism satisfies rho-zCDP."""
  # rho = 1/8 * epsilon^2
  return math.sqrt(8 * rho)


def gdp_gaussian_sigma(budget: float) -> float:
  """Return the Gaussian mechanism sigma that satisfies `budget`-GDP."""
  return math.sqrt(1.0 / budget)


def gdp_bounded_range_musq(nu: float) -> float:
  """Return the squared GDP parameter mu^2 of a bounded range mechanism.

  A mechanism with bounded range parameter nu satisfies mu-GDP for
  mu = -2 * Phi^{-1}(1 / (exp(nu / 2) + 1)).

  Args:
    nu: The bounded range parameter of the mechanism.
  """
  assert nu >= 0
  mu = -2.0 * scipy.stats.norm.ppf(1.0 / (math.exp(nu / 2.0) + 1.0))
  return mu**2


def gdp_bounded_range_nu(musq: float) -> float:
  """Return the largest bounded range parameter nu that satisfies mu-GDP.

  This is the inverse of `gdp_bounded_range_musq`, given by nu = 2 * L(mu)
  with L(t) = log(Phi(t / 2) / Phi(-t / 2)).

  Args:
    musq: The GDP budget mu^2 to spend on the bounded range mechanism.
  """
  assert musq >= 0
  mu = math.sqrt(musq)
  denom = scipy.stats.norm.cdf(-mu / 2.0)
  if denom == 0.0:
    return 2.0 * (
        scipy.special.log_ndtr(mu / 2.0) - scipy.special.log_ndtr(-mu / 2.0)
    )
  return 2.0 * math.log(scipy.stats.norm.cdf(mu / 2.0) / denom)


def gdp_exponential_eps(budget: float) -> float:
  """Return the exponential mechanism epsilon that satisfies `budget`-GDP."""
  return gdp_bounded_range_nu(budget)


def zcdp_to_gdp(rho: float) -> float:
  """Return the largest GDP budget (mu^2) such that mu-GDP implies rho-zCDP."""
  # rho = 0.5 / sigma^2 = 0.5 * budget
  return 2 * rho

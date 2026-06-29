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

"""Deprecated shim.

Use :class:`dpsynth.data_generation_v3.TabularSynthesizer` instead.
"""

from collections.abc import Mapping, Sequence
import warnings

from dpsynth import constraints
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.data_generation_v3 import TabularSynthesizer
import numpy as np
import pandas as pd


def generate(
    data: pd.DataFrame,
    domains: Mapping[str, domain.AttributeType],
    epsilon: float,
    delta: float,
    *,
    discrete_config: (
        discrete_mechanisms.DiscreteMechanism
    ) = discrete_mechanisms.MSTMechanism(),
    numerical_bins: int = 32,
    one_way_marginal_budget_fraction: float = 0.1,
    cross_attribute_constraints: Sequence[constraints.Constraint] = (),
    skip_compression: bool = False,
) -> pd.DataFrame:
  """Deprecated. Use :class:`data_generation_v3.TabularSynthesizer` instead."""
  warnings.warn(
      'data_generation_v2.generate() is deprecated. Use'
      ' data_generation_v3.TabularSynthesizer instead.',
      DeprecationWarning,
      stacklevel=2,
  )
  del skip_compression  # Not supported by TabularSynthesizer.
  synth = TabularSynthesizer(
      domains=domains,
      discrete_mechanism=discrete_config,
      cross_attribute_constraints=cross_attribute_constraints,
  )
  result = synth.calibrate(
      epsilon=epsilon,
      delta=delta,
      numerical_bins=numerical_bins,
      init_budget_fraction=one_way_marginal_budget_fraction,
  )(np.random.default_rng(), data)
  return result.synthetic_data

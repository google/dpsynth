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

"""Differentially private synthesizer for nested/typed tabular data.

Privacy model — neighboring relation and participation assumptions:

  Each individual contributes exactly one row to exactly one type-table.
  Two datasets are *neighbors* if they differ in the addition or removal
  of a single such row.  Because a row appears in only one type-table,
  the per-type detail models operate on disjoint partitions and satisfy
  parallel composition: the privacy cost of the detail level equals the
  cost of a single type, not the sum over types.
"""

from __future__ import annotations

import dataclasses

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth import data_generation_v3
from dpsynth import discrete_mechanisms
from dpsynth import domain
import numpy as np
import pandas as pd


@dataclasses.dataclass
class NestedSynthesisResult:
  """Synthetic DataFrames keyed by type name, mirroring the input structure."""

  synthetic_data: dict[str, pd.DataFrame]


def _resample_to_size(rng, df, n):
  """Shuffles df and tiles it cyclically to exactly n rows."""
  m = len(df)
  if m == n:
    return df.reset_index(drop=True)
  shuffled = df.sample(frac=1, random_state=rng.integers(2**31)).reset_index(
      drop=True
  )
  indices = np.arange(n) % m
  return shuffled.iloc[indices].reset_index(drop=True)


def _stitch_synthetic_output(
    rng,
    synthetic_shared,
    synthetic_details,
    type_vocabulary,
):
  """Stitches shared and per-type synthetic attributes into typed records."""
  output: dict[str, pd.DataFrame] = {}
  for type_name in type_vocabulary:
    type_mask = synthetic_shared['_type'] == type_name
    type_shared = synthetic_shared.loc[type_mask].drop(columns=['_type'])
    n_synthetic = len(type_shared)
    if n_synthetic == 0:
      continue

    type_shared = type_shared.reset_index(drop=True)
    if type_name in synthetic_details:
      detail = synthetic_details[type_name]
      detail = _resample_to_size(rng, detail, n_synthetic)
      # Assumes conditional independence: P(detail | type, shared) ≈
      # P(detail | type). Could be tightened with shared alignment keys.
      output[type_name] = pd.concat([type_shared, detail], axis=1)
    else:
      output[type_name] = type_shared

  logging.info('[DPSynth Nested]: Generated records for %d types.', len(output))
  return NestedSynthesisResult(synthetic_data=output)


@dataclasses.dataclass
class NestedTabularSynthesizer(api.DPMechanism):
  """DP synthesizer for datasets with typed records.

  Each record has a type and type-specific attributes. Different types may
  have different schemas. The synthesizer learns:

  1. **Shared model (Model 1):** Joint distribution over record types and
     shared attributes using a TabularSynthesizer.
  2. **Per-type detail models (Model 2):** Independent TabularSynthesizer
     per type for type-specific attributes, under parallel composition.

  Privacy cost is: sequential composition of Model 1 + Model 2. Model 2
  costs nothing extra under parallel composition (each type's data is
  disjoint), so only one representative type's cost is counted.

  Example:

  >>> import numpy as np
  >>> import pandas as pd
  >>> from dpsynth import domain
  >>> from dpsynth.experimental import nested
  >>> Cat = domain.CategoricalAttribute
  >>> synth = nested.NestedTabularSynthesizer(
  ...     shared_schema={'platform': Cat(['web', 'mobile'])},
  ...     per_type_schemas={
  ...         'click': {'element': Cat(['button', 'link'])},
  ...         'purchase': {'amount': Cat(['low', 'high'])},
  ...     },
  ... )
  >>> calibrated = synth.configure(zcdp_rho=1.0)
  >>> rng = np.random.default_rng(42)
  >>> rows = {'platform': ['web', 'mobile'] * 20}
  >>> click_df = pd.DataFrame({**rows, 'element': ['button', 'link'] * 20})
  >>> purchase_df = pd.DataFrame({**rows, 'amount': ['low', 'high'] * 20})
  >>> data = {'click': click_df, 'purchase': purchase_df}
  >>> result = calibrated(rng, data)  # doctest: +SKIP
  >>> sorted(result.synthetic_data)  # doctest: +SKIP
  ['click', 'purchase']

  Attributes:
    shared_schema: Domains for columns shared across all record types.
    per_type_schemas: Type-specific attribute domains. Keys define the type
      vocabulary. Types not present get empty attribute dicts.
    shared_budget_fraction: Fraction of total rho for Model 1.
    shared_mechanism: Discrete mechanism for the shared model.
    detail_mechanism: Discrete mechanism for per-type models.
    init_budget_fraction: Within each TabularSynthesizer, fraction for
      initializers.
  """

  shared_schema: domain.Schema
  per_type_schemas: dict[str, domain.Schema]
  shared_budget_fraction: float = 0.5
  shared_mechanism: discrete_mechanisms.DiscreteMechanism = dataclasses.field(
      default_factory=discrete_mechanisms.MSTMechanism
  )
  detail_mechanism: discrete_mechanisms.DiscreteMechanism = dataclasses.field(
      default_factory=discrete_mechanisms.MSTMechanism
  )
  init_budget_fraction: float = 0.1

  # Internal state set by configure().
  _shared_synth: data_generation_v3.TabularSynthesizer | None = (
      dataclasses.field(default=None, repr=False)
  )
  _detail_synths: dict[str, data_generation_v3.TabularSynthesizer] | None = (
      dataclasses.field(default=None, repr=False)
  )
  _detail_rho: float | None = dataclasses.field(default=None, repr=False)

  @property
  def type_vocabulary(self) -> list[str]:
    """Returns the list of type names from per_type_schemas keys."""
    return list(self.per_type_schemas.keys())

  def configure(
      self, *, zcdp_rho: float, delta: float = 0.0, **kwargs
  ) -> NestedTabularSynthesizer:
    """Returns a configured copy with the given zCDP budget."""
    # Additive zCDP split; each type gets full rho_detail
    # (parallel composition over disjoint type partitions).
    rho_shared = self.shared_budget_fraction * zcdp_rho
    rho_detail = (1 - self.shared_budget_fraction) * zcdp_rho

    shared_domains = dict(self.shared_schema)
    shared_domains['_type'] = domain.CategoricalAttribute(
        possible_values=self.type_vocabulary
    )
    shared_synth = data_generation_v3.TabularSynthesizer(
        domains=shared_domains,
        discrete_mechanism=self.shared_mechanism,
        init_budget_fraction=self.init_budget_fraction,
    ).configure(zcdp_rho=rho_shared)

    detail_synths = {}
    for type_name, type_schema in self.per_type_schemas.items():
      if not type_schema:
        continue  # No type-specific attributes.
      detail_synths[type_name] = data_generation_v3.TabularSynthesizer(
          domains=dict(type_schema),
          discrete_mechanism=self.detail_mechanism,
          init_budget_fraction=self.init_budget_fraction,
      ).configure(zcdp_rho=rho_detail)

    return dataclasses.replace(
        self,
        _shared_synth=shared_synth,
        _detail_synths=detail_synths,
        _detail_rho=rho_detail,
    )

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed DpEvent for the full mechanism."""
    # supports it, instead of falling back to a ZCDpEvent.
    if self._shared_synth is None:
      raise ValueError(
          'Must call configure() or calibrate() before accessing dp_event.'
      )
    events = [self._shared_synth.dp_event]
    if self._detail_rho is not None and self._detail_rho > 0:
      # Report a ZCDpEvent rather than an arbitrary type's dp_event because
      # per-type events may differ (different schemas) and picking one could
      # undercount under PLD accounting.
      events.append(dp_accounting.ZCDpEvent(self._detail_rho))
    return dp_accounting.ComposedDpEvent(events)

  def __call__(
      self,
      rng: np.random.Generator,
      data: dict[str, pd.DataFrame],
  ) -> NestedSynthesisResult:
    """Runs the mechanism on per-type DataFrames, returns synthetic records."""
    if self._shared_synth is None or self._detail_synths is None:
      raise ValueError('Must call configure() or calibrate() before running.')

    shared_cols = list(self.shared_schema)

    shared_df = pd.concat(
        [data[t][shared_cols].assign(_type=t) for t in data],
        ignore_index=True,
    )
    synthetic_shared = self._shared_synth(rng, shared_df).synthetic_data

    synthetic_details: dict[str, pd.DataFrame] = {}
    for type_name, synth in self._detail_synths.items():
      if type_name in data:
        detail_cols = list(self.per_type_schemas[type_name])
        synthetic_details[type_name] = synth(
            rng, data[type_name][detail_cols]
        ).synthetic_data

    return _stitch_synthetic_output(
        rng, synthetic_shared, synthetic_details, self.type_vocabulary
    )

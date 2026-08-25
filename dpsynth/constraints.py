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

"""Cross-attribute constraints for categorical data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from dpsynth import domain
from dpsynth import transformations
import mbi
import numpy as np


def _validate(c: Constraint) -> None:
  """Validate a Constraint's fields."""
  if c.attribute_domains is not None and len(c.attribute_names) != len(
      c.attribute_domains
  ):
    raise ValueError(
        'attribute_names and attribute_domains must have the same length, got'
        f' {len(c.attribute_names)} != {len(c.attribute_domains)}.'
    )
  modes = (
      c.possible_combinations,
      c.impossible_combinations,
      c.functional_dependency,
  )
  n_set = sum(x is not None for x in modes)
  if n_set != 1:
    raise ValueError(
        "Specify exactly one of 'possible_combinations',"
        " 'impossible_combinations', or 'functional_dependency'."
    )
  if c.functional_dependency is not None and len(c.attribute_names) != 2:
    raise ValueError(
        'functional_dependency requires exactly 2 attributes (fine, coarse),'
        f' got {len(c.attribute_names)}.'
    )
  if c.functional_dependency is None:
    combos = c.possible_combinations or c.impossible_combinations
    n_attrs = len(c.attribute_names)
    for combo in combos:  # pyrefly: ignore[not-iterable]
      if len(combo) != n_attrs:
        raise ValueError(
            'Each combination must have length equal to the number of'
            f' attributes ({n_attrs}), got {len(combo)}.'
        )


@dataclasses.dataclass(frozen=True)
class Constraint:
  """A constraint on allowed value combinations across attributes.

  Mirrors :class:`mbi.Constraint` but accepts human-readable values from
  :class:`dpsynth.domain.CategoricalAttribute` instead of integer arrays.
  Exactly one of ``possible_combinations``, ``impossible_combinations``, or
  ``functional_dependency`` must be specified.

  Example Usage:

    >>> d1 = domain.CategoricalAttribute(['GameSuite', 'OfficePro', 'DevTool'])
    >>> d2 = domain.CategoricalAttribute(['Windows', 'Linux', 'MacOS'])
    >>> constraint = Constraint(
    ...     attribute_names=('Software', 'Operating System'),
    ...     attribute_domains=(d1, d2),
    ...     possible_combinations=[
    ...         ('GameSuite', 'Windows'),
    ...         ('OfficePro', 'Windows'),
    ...         ('OfficePro', 'MacOS'),
    ...         ('DevTool', 'Linux'),
    ...         ('DevTool', 'MacOS'),
    ...     ],
    ... )

  Attributes:
    attribute_names: Names of the constrained attributes.
    attribute_domains: Categorical domain for each attribute. Optional when
      associated with a Schema or provided to ``to_mbi(schema)``.
    possible_combinations: Allowed value combinations.
    impossible_combinations: Forbidden value combinations.
    functional_dependency: Dict mapping fine attribute values to coarse
      attribute values. Requires exactly two attributes.
  """

  attribute_names: tuple[str, ...]
  attribute_domains: tuple[domain.CategoricalAttribute, ...] | None = None
  possible_combinations: Sequence[tuple[Any, ...]] | None = None
  impossible_combinations: Sequence[tuple[Any, ...]] | None = None
  functional_dependency: Mapping[Any, Any] | None = None

  def __post_init__(self):
    _validate(self)

  def bind_schema(
      self, schema: Mapping[str, domain.AttributeType]
  ) -> Constraint:
    """Returns a new Constraint with attribute_domains resolved from schema."""
    if self.attribute_domains is not None:
      return self
    resolved_domains = []
    for name in self.attribute_names:
      if name not in schema:
        raise KeyError(
            f"Attribute '{name}' from constraint not found in schema."
        )
      attr = schema[name]
      if not isinstance(attr, domain.CategoricalAttribute):
        raise TypeError(
            f"Constraint attribute '{name}' must be CategoricalAttribute, got"
            f' {type(attr).__name__}.'
        )
      resolved_domains.append(attr)
    return dataclasses.replace(self, attribute_domains=tuple(resolved_domains))

  def to_mbi(
      self, schema: Mapping[str, domain.AttributeType] | None = None
  ) -> mbi.Constraint:
    """Convert to an mbi.Constraint."""
    bound = self.bind_schema(schema) if schema is not None else self
    if bound.attribute_domains is None:
      raise ValueError(
          'Constraint has no attribute_domains; provide a schema to to_mbi() or'
          ' construct with attribute_domains.'
      )

    shape = tuple(d.size for d in bound.attribute_domains)
    mbi_domain = mbi.Domain(bound.attribute_names, shape)
    encoders = [
        transformations.discrete_encoder(d) for d in bound.attribute_domains
    ]

    if bound.functional_dependency is not None:
      _, coarse_enc = encoders
      fine_values = bound.attribute_domains[0].possible_values
      coarse_indices = [
          coarse_enc(bound.functional_dependency[v]) for v in fine_values
      ]
      return mbi.Constraint(
          domain=mbi_domain, mapping=np.array(coarse_indices, dtype=np.int32)
      )

    combos = bound.possible_combinations or bound.impossible_combinations
    encoded = [[enc(v) for enc, v in zip(encoders, c)] for c in combos]  # pyrefly: ignore[not-iterable]
    indices = np.array(encoded, dtype=np.int32)
    if bound.possible_combinations is not None:
      return mbi.Constraint(domain=mbi_domain, valid=indices)
    return mbi.Constraint(domain=mbi_domain, invalid=indices)

  def to_dict(self) -> dict[str, Any]:
    """Converts the Constraint into a serializable dictionary."""
    data: dict[str, Any] = {'attribute_names': list(self.attribute_names)}
    if self.possible_combinations is not None:
      data['possible_combinations'] = [
          list(c) for c in self.possible_combinations
      ]
    if self.impossible_combinations is not None:
      data['impossible_combinations'] = [
          list(c) for c in self.impossible_combinations
      ]
    if self.functional_dependency is not None:
      data['functional_dependency'] = dict(self.functional_dependency)
    if self.attribute_domains is not None:
      data['attribute_domains'] = [
          domain.attribute_to_dict(d) for d in self.attribute_domains
      ]
    return data

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> Constraint:
    """Instantiates a Constraint from a dictionary."""
    attr_names = tuple(data['attribute_names'])
    attr_domains = None
    if 'attribute_domains' in data and data['attribute_domains'] is not None:
      parsed_domains = []
      for d in data['attribute_domains']:
        parsed = domain.attribute_from_dict(d)
        if not isinstance(parsed, domain.CategoricalAttribute):
          raise TypeError(
              'Constraint attribute_domains must be CategoricalAttribute, got'
              f' {type(parsed).__name__}.'
          )
        parsed_domains.append(parsed)
      attr_domains = tuple(parsed_domains)
    possible_combinations = None
    if (
        'possible_combinations' in data
        and data['possible_combinations'] is not None
    ):
      possible_combinations = [tuple(c) for c in data['possible_combinations']]
    impossible_combinations = None
    if (
        'impossible_combinations' in data
        and data['impossible_combinations'] is not None
    ):
      impossible_combinations = [
          tuple(c) for c in data['impossible_combinations']
      ]
    functional_dependency = data.get('functional_dependency')
    return cls(
        attribute_names=attr_names,
        attribute_domains=attr_domains,
        possible_combinations=possible_combinations,
        impossible_combinations=impossible_combinations,
        functional_dependency=functional_dependency,
    )

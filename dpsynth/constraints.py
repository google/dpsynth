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

"""Helpers for defining and validating cross-attribute constraints."""

from collections.abc import Iterable, Sequence
import dataclasses
import functools
from typing import Any

from dpsynth import domain
from dpsynth import transformations
import jax.numpy as jnp
import mbi


@dataclasses.dataclass(frozen=True)
class Constraint:
  """Class for specifying cross-attribute constraints for categorical data.

  For technical details, see Section D of https://arxiv.org/pdf/2201.12677.

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
    ...         ('DevTool', 'MacOS')
    ...     ]
    ... )
  """

  attribute_names: tuple[str, ...]
  attribute_domains: tuple[domain.CategoricalAttribute, ...]
  possible_combinations: Iterable[tuple[Any, ...]]

  def __post_init__(self):
    if len(self.attribute_names) != len(self.attribute_domains):
      raise ValueError(
          'attribute_names and attribute_domains must have the same length, got'
          f' {len(self.attribute_names)} != {len(self.attribute_domains)}.'
      )
    for combination in self.possible_combinations:
      if len(combination) != len(self.attribute_names):
        raise ValueError(
            'Each combination must have length equal to the number of '
            f'attributes, got {len(combination), len(self.attribute_names)}.'
        )

      for i, (name, value) in enumerate(zip(self.attribute_names, combination)):
        dom = self.attribute_domains[i]
        if value not in dom.possible_values:
          raise ValueError(
              f'Value {value} for attribute {name} not found in possible'
              f' values {dom.possible_values}.'
          )


def _possible_indices(constraint: Constraint) -> tuple[tuple[int, ...], ...]:
  """Returns the possible indices of the constraint."""
  c = constraint
  transform_fns = [
      transformations.discrete_encoder(dom) for dom in c.attribute_domains
  ]

  def fun(values):
    return tuple(transform_fns[i](x) for i, x in enumerate(values))

  return tuple(fun(combination) for combination in c.possible_combinations)


def _mbi_domain(constraint: Constraint) -> mbi.Domain:
  """Returns the domain for a constraint."""
  shape = tuple(dom.size for dom in constraint.attribute_domains)
  return mbi.Domain(constraint.attribute_names, shape)


def _mbi_potential(constraint: Constraint) -> mbi.Factor:
  """Returns the potential domain for a constraint."""
  idx = tuple(zip(*_possible_indices(constraint)))
  mbi_domain = _mbi_domain(constraint)
  values = jnp.full(mbi_domain.shape, -jnp.inf).at[idx].set(0)
  return mbi.Factor(mbi_domain, values)


def get_initial_parameters(
    constraints: Sequence[Constraint],
    dom: mbi.Domain | None = None,
) -> mbi.CliqueVector:
  """Returns initial markov random field parameters for the constraints."""
  parameters = {c.attribute_names: _mbi_potential(c) for c in constraints}
  cliques = [c.attribute_names for c in constraints]
  if dom is None:
    dom = functools.reduce(
        lambda d1, d2: d1.merge(d2), [_mbi_domain(c) for c in constraints]
    )
  return mbi.CliqueVector(dom, cliques, parameters)

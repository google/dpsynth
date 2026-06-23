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

"""Synthetic Tabular Data API for synthesizing collections of pydantic Models."""

from collections.abc import Iterable
import enum
import inspect
import math
import types
import typing  # for typing.get_origin and typing.get_args
from typing import Any, Literal, TypeVar

from dpsynth import data_generation_v2
from dpsynth import discrete_mechanisms
from dpsynth import domain
import pandas as pd
import pydantic
from pydantic.fields import annotated_types
from pydantic.fields import FieldInfo  # pylint: disable=g-importing-member


def _get_base_type(annotation: type[Any]) -> tuple[bool, type[Any]]:
  """Determines the base type of a type annotation and if it is optional."""

  if annotation is types.NoneType:
    raise ValueError("Unexpected None type annotation.")

  is_optional = annotation == (annotation | None)

  if is_optional:
    # It's a Union like `int | None`. We need the non-None part.
    args = typing.get_args(annotation)
    non_none_args = [arg for arg in args if arg is not types.NoneType]
    assert len(non_none_args) == 1, "Union must have exactly one non-None type."
    annotation = non_none_args[0]

  if not (
      annotation in (int, float, str, bool)
      or (inspect.isclass(annotation) and issubclass(annotation, enum.Enum))
      or typing.get_origin(annotation) is Literal
  ):
    raise ValueError(f"Unexpected type annotation: {annotation}.")

  return is_optional, annotation


def _numerical_attribute_from_field_info(
    field_info: FieldInfo,
) -> domain.NumericalAttribute:
  """Infers a NumericalAttribute from a pydantic FieldInfo."""
  # NumericalAttribute uses a convention where both the min_value and max_value
  # are assumed to be inclusive.  More general bounds are supported by the
  # pydantic metadata, so we convert to the expected representation here.
  optional, base_type = _get_base_type(field_info.annotation)

  lower_bound = upper_bound = None
  for meta in field_info.metadata:
    match type(meta):
      case annotated_types.Ge:
        lower_bound = meta.ge
      case annotated_types.Le:
        upper_bound = meta.le
      case annotated_types.Gt:
        if base_type is int:
          lower_bound = meta.gt + 1
        else:
          lower_bound = math.nextafter(meta.gt, math.inf)
      case annotated_types.Lt:
        if base_type is int:
          upper_bound = meta.lt - 1
        else:
          upper_bound = math.nextafter(meta.lt, -math.inf)
      case _:
        continue
  if lower_bound is None or upper_bound is None:
    raise ValueError("Must specify lower and upper bounds for numeric fields.")

  return domain.NumericalAttribute(
      min_value=lower_bound,
      max_value=upper_bound,
      clip_to_range=not optional,
      dtype=base_type.__name__,
  )


def _categorical_attribute_from_field_info(
    field_info: FieldInfo,
) -> domain.CategoricalAttribute:
  """Infers a CategoricalAttribute from a pydantic FieldInfo."""
  optional, base_type = _get_base_type(field_info.annotation)
  if inspect.isclass(base_type) and issubclass(base_type, enum.Enum):
    possible_values = list(base_type)
  elif typing.get_origin(base_type) is Literal:
    possible_values = list(typing.get_args(base_type))
  elif base_type is bool:
    possible_values = [False, True]
  else:
    raise ValueError(f"Unexpected type annotation: {base_type}.")

  if optional:
    possible_values = [None] + possible_values

  return domain.CategoricalAttribute(
      possible_values=possible_values, out_of_domain_index=0
  )


def infer_domain_from_model(
    model_cls: type[pydantic.BaseModel],
) -> dict[str, domain.CategoricalAttribute | domain.NumericalAttribute]:
  """Infers the domain of a pydantic model."""

  attributes = {}
  for name, meta in model_cls.model_fields.items():
    _, base_type = _get_base_type(meta.annotation)
    if base_type in (int, float):
      attributes[name] = _numerical_attribute_from_field_info(meta)
    elif base_type is bool:
      attributes[name] = _categorical_attribute_from_field_info(meta)
    elif inspect.isclass(base_type) and issubclass(base_type, enum.Enum):
      attributes[name] = _categorical_attribute_from_field_info(meta)
    elif typing.get_origin(base_type) is Literal:
      attributes[name] = _categorical_attribute_from_field_info(meta)
    else:
      raise ValueError(f"Unexpected type annotation: {base_type}.")
  return attributes


RecordT = TypeVar("RecordT", bound=pydantic.BaseModel)


def dp_synthetic_data_generation(
    data: Iterable[RecordT],
    epsilon: float,
    delta: float,
    *,
    mechanism_config: discrete_mechanisms.DiscreteMechanism = discrete_mechanisms.MSTMechanism(),
) -> list[RecordT]:
  """Generate synthetic data for a collection of pydantic Models.

  Args:
    data: The collection of pydantic Models to generate synthetic data for.
    epsilon: Privacy parameter.
    delta: Privacy parameter.
    mechanism_config: The algorithm configuration to use for synthetic data
      generation. Defaults to MST.

  Returns:
    A synthetic collection of records of the same type as the input, with
    synthetic values for each field that match the original data in aggregate.
  """
  data = list(data)
  cls = data[0].__class__

  synthetic = data_generation_v2.generate(
      data=pd.DataFrame([user.model_dump() for user in data], dtype="object"),
      domains=infer_domain_from_model(cls),
      epsilon=epsilon,
      delta=delta,
      discrete_config=mechanism_config,
  )

  return [
      cls(**{
          k: None if isinstance(v, float) and math.isnan(v) else v
          for k, v in user.items()
      })
      for _, user in synthetic.iterrows()
  ]

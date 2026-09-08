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
"""Protobuf descriptor to dpsynth domain adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from dpsynth import domain
from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import message_factory

OpenSetAttribute = domain.OpenSetCategoricalAttribute

_INTEGER_TYPES = {
    descriptor.FieldDescriptor.TYPE_INT32,
    descriptor.FieldDescriptor.TYPE_INT64,
    descriptor.FieldDescriptor.TYPE_UINT32,
    descriptor.FieldDescriptor.TYPE_UINT64,
    descriptor.FieldDescriptor.TYPE_SINT32,
    descriptor.FieldDescriptor.TYPE_SINT64,
    descriptor.FieldDescriptor.TYPE_FIXED32,
    descriptor.FieldDescriptor.TYPE_FIXED64,
    descriptor.FieldDescriptor.TYPE_SFIXED32,
    descriptor.FieldDescriptor.TYPE_SFIXED64,
}

_FLOAT_TYPES = {
    descriptor.FieldDescriptor.TYPE_FLOAT,
    descriptor.FieldDescriptor.TYPE_DOUBLE,
}


def _is_repeated(field) -> bool:
  return (
      getattr(field, "is_repeated", False)
      or getattr(field, "label", None)
      == descriptor.FieldDescriptor.LABEL_REPEATED
  )


def _resolve_descriptor(proto):
  """Resolves input to a protobuf Message Descriptor."""
  if isinstance(proto, descriptor.Descriptor):
    return proto
  desc = getattr(proto, "DESCRIPTOR", None)
  if isinstance(desc, descriptor.Descriptor):
    return desc
  raise TypeError(
      "Expected a Protobuf Message class, instance, or Descriptor, got"
      f" {type(proto)}."
  )


def infer_domain(
    proto: descriptor.Descriptor | type[message.Message] | message.Message,
    *,
    numerical_bounds: Mapping[str, tuple[float, float]] | None = None,
    enum_format: Literal["name", "number"] = "name",
    ignore_unsupported_fields: bool = False,
) -> dict[str, domain.AttributeType]:
  """Infers attribute domains from a Protobuf definition for a flat schema."""
  if enum_format not in ("name", "number"):
    raise ValueError(f"Unknown enum_format '{enum_format}'.")

  msg_desc = _resolve_descriptor(proto)
  attributes: dict[str, domain.AttributeType] = {}
  numerical_bounds = numerical_bounds or {}

  for field in msg_desc.fields:
    if _is_repeated(field):
      if ignore_unsupported_fields:
        continue
      raise ValueError(f"Repeated field '{field.name}' is not supported.")

    if field.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
      # To extend to nested schemas, message fields could be recursively
      # flattened with delimited column names or modeled as linked sub-tables.
      if ignore_unsupported_fields:
        continue
      raise ValueError(
          f"Nested message field '{field.name}' ({field.message_type.name}) is"
          " not supported in flat schema."
      )

    if field.type == descriptor.FieldDescriptor.TYPE_ENUM:
      values = [
          v.name if enum_format == "name" else v.number
          for v in field.enum_type.values
      ]
      attributes[field.name] = domain.CategoricalAttribute(
          possible_values=values, out_of_domain_index=0
      )
    elif field.type in _INTEGER_TYPES or field.type in _FLOAT_TYPES:
      dtype = "int" if field.type in _INTEGER_TYPES else "float"
      if field.name not in numerical_bounds:
        raise ValueError(
            f"Numerical bounds must be specified for field '{field.name}'."
        )
      bounds = numerical_bounds[field.name]
      attributes[field.name] = domain.NumericalAttribute(
          min_value=bounds[0], max_value=bounds[1], dtype=dtype
      )
    elif field.type == descriptor.FieldDescriptor.TYPE_STRING:
      attributes[field.name] = domain.OpenSetCategoricalAttribute()
    elif field.type == descriptor.FieldDescriptor.TYPE_BOOL:
      attributes[field.name] = domain.CategoricalAttribute(
          possible_values=[False, True], out_of_domain_index=0
      )
    else:
      if ignore_unsupported_fields:
        continue
      raise ValueError(
          f"Field '{field.name}' of type {field.type} is not supported."
      )

  return attributes


def infer_schema(
    proto: descriptor.Descriptor | type[message.Message] | message.Message,
    *,
    numerical_bounds: Mapping[str, tuple[float, float]] | None = None,
    enum_format: Literal["name", "number"] = "name",
    ignore_unsupported_fields: bool = False,
) -> domain.Schema:
  """Derives a dpsynth.Schema from a Protobuf definition for a flat schema."""
  return domain.Schema(
      infer_domain(
          proto,
          numerical_bounds=numerical_bounds,
          enum_format=enum_format,
          ignore_unsupported_fields=ignore_unsupported_fields,
      )
  )


def to_tuple(
    msg: message.Message,
    *,
    enum_format: Literal["name", "number"] = "name",
) -> tuple[Any, ...]:
  """Converts a flat Protobuf message instance to a tuple of field values."""
  result = []
  for field in msg.DESCRIPTOR.fields:
    val = getattr(msg, field.name)
    if (
        field.type == descriptor.FieldDescriptor.TYPE_ENUM
        and enum_format == "name"
    ):
      val = field.enum_type.values_by_number[val].name
    result.append(val)
  return tuple(result)


def from_tuple(
    values: Sequence[Any],
    proto: descriptor.Descriptor | type[message.Message] | message.Message,
) -> message.Message:
  """Populates a flat Protobuf message from a sequence of field values."""
  desc = _resolve_descriptor(proto)
  msg = message_factory.GetMessageClass(desc)()
  for field, val in zip(desc.fields, values):
    setattr(msg, field.name, val)
  return msg


infer_domain_from_proto = infer_domain
schema_from_proto = infer_schema

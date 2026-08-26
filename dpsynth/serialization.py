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

"""General YAML serialization and deserialization for DPSynth dataclasses."""

from collections.abc import Mapping
import dataclasses
import importlib
import inspect
from typing import Any

from absl import logging
from dpsynth import api
from dpsynth import domain
from etils import epath
import yaml

PathType = epath.PathLike


def _ensure_subclasses_loaded() -> None:
  """Imports modules containing MechanismConfig subclasses to register them."""
  modules = [
      'dpsynth.data_generation_v3',
      'dpsynth.discrete_mechanisms.aim',
      'dpsynth.discrete_mechanisms.aim_gdp',
      'dpsynth.discrete_mechanisms.direct',
      'dpsynth.discrete_mechanisms.discrete',
      'dpsynth.discrete_mechanisms.independent',
      'dpsynth.discrete_mechanisms.mst',
      'dpsynth.discrete_mechanisms.swift',
      'dpsynth.relational.synthesizer',
      'dpsynth.local_mode.initialization',
  ]
  for mod_name in modules:
    try:
      importlib.import_module(mod_name)
    except (ImportError, AttributeError) as e:
      logging.debug('Could not import %s: %s', mod_name, e)


def _get_foreign_key_relation_cls() -> type[Any] | None:
  """Lazily resolves ForeignKeyRelation class if available."""
  try:
    mod = importlib.import_module('dpsynth.relational.domain')
    return getattr(mod, 'ForeignKeyRelation', None)
  except (ImportError, AttributeError):
    return None


def _get_registered_classes() -> dict[str, type[Any]]:
  """Discovers all serializable dataclasses across dpsynth dynamically."""
  _ensure_subclasses_loaded()
  classes: dict[str, type[Any]] = {}

  def _collect(cls: type[Any]) -> None:
    for sub in cls.__subclasses__():
      classes[sub.__name__] = sub
      _collect(sub)

  _collect(api.MechanismConfig)

  for cls in (
      domain.Schema,
      domain.CategoricalAttribute,
      domain.NumericalAttribute,
      domain.OpenSetCategoricalAttribute,
      domain.FreeFormTextAttribute,
  ):
    classes[cls.__name__] = cls

  fk_cls = _get_foreign_key_relation_cls()
  if fk_cls is not None:
    classes[fk_cls.__name__] = fk_cls

  return classes


def to_dict(obj: Any) -> Any:
  """Recursively converts dataclasses, mappings, sequences, and primitives to dicts.

  Args:
    obj: The object to serialize (e.g. MechanismConfig, Attribute, Schema).

  Returns:
    A JSON/YAML-serializable Python data structure.
  """
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    data = {'type': obj.__class__.__name__}
    for field in dataclasses.fields(obj):
      data[field.name] = to_dict(getattr(obj, field.name))
    return data
  elif isinstance(obj, Mapping):
    return {k: to_dict(v) for k, v in obj.items()}
  elif isinstance(obj, (list, tuple)):
    return [to_dict(v) for v in obj]
  else:
    return obj


def from_dict(data: Any, expected_cls: type[Any] | None = None) -> Any:
  """Recursively converts dictionaries with a 'type' field into dataclass instances.

  Args:
    data: Data structure from YAML safe_load.
    expected_cls: Optional expected target class.

  Returns:
    Reconstructed dataclass instance, dict, list, or primitive.

  Raises:
    ValueError: If a dictionary contains an unknown 'type' or cannot be
    resolved.
  """
  if isinstance(data, dict):
    if 'type' in data:
      type_name = data['type']
      classes = _get_registered_classes()
      target_cls = classes.get(type_name)
      if target_cls is None:
        raise ValueError(f"Unknown type in YAML: '{type_name}'")
      if (
          expected_cls is not None
          and isinstance(expected_cls, type)
          and not issubclass(target_cls, expected_cls)
      ):
        raise ValueError(
            f'Expected {expected_cls.__name__}, got {target_cls.__name__}'
        )

      kwargs = {}
      for field in dataclasses.fields(target_cls):
        if field.name in data:
          val = from_dict(data[field.name])
          if isinstance(val, list):
            # Normalize list of lists to list of tuples (e.g. marginal query cliques)
            val = [tuple(x) if isinstance(x, list) else x for x in val]
            if isinstance(field.default, tuple):
              val = tuple(val)
          kwargs[field.name] = val
      return target_cls(**kwargs)
    elif (
        expected_cls is not None
        and dataclasses.is_dataclass(expected_cls)
        and not inspect.isabstract(expected_cls)
    ):
      kwargs = {}
      for field in dataclasses.fields(expected_cls):
        if field.name in data:
          kwargs[field.name] = from_dict(data[field.name])
      return expected_cls(**kwargs)
    elif expected_cls is not None and inspect.isabstract(expected_cls):
      raise ValueError(
          f"Missing 'type' field in dictionary for {expected_cls.__name__}"
      )
    else:
      return {k: from_dict(v) for k, v in data.items()}
  elif isinstance(data, list):
    return [from_dict(v) for v in data]
  else:
    return data


def to_yaml(obj: Any) -> str:
  """Serializes an object into a YAML string.

  Args:
    obj: The object to serialize.

  Returns:
    A YAML string representation of the object.
  """
  return yaml.dump(to_dict(obj), default_flow_style=False, sort_keys=False)


def to_yaml_file(obj: Any, filepath: str | PathType) -> None:
  """Writes an object to a YAML file.

  Args:
    obj: The object to serialize.
    filepath: Destination file path.
  """
  path = epath.Path(filepath)
  path.write_text(to_yaml(obj))


def from_yaml(yaml_str: str, expected_cls: type[Any] | None = None) -> Any:
  """Loads an object from a YAML string.

  Args:
    yaml_str: YAML string encoding an object.
    expected_cls: Optional expected target class.

  Returns:
    The reconstructed object instance.
  """
  yaml_data = yaml.safe_load(yaml_str)
  if not isinstance(yaml_data, dict):
    raise ValueError(
        f'YAML root must be a mapping, got {type(yaml_data).__name__}'
    )
  if expected_cls is None and 'type' not in yaml_data:
    raise ValueError(
        "Missing 'type' field at root of YAML. All DPSynth serialized YAMLs"
        " must specify 'type'."
    )
  return from_dict(yaml_data, expected_cls=expected_cls)


def from_yaml_file(
    filepath: str | PathType, expected_cls: type[Any] | None = None
) -> Any:
  """Loads an object from a YAML file.

  Args:
    filepath: Path to the YAML file.
    expected_cls: Optional expected target class.

  Returns:
    The reconstructed object instance.
  """
  path = epath.Path(filepath)
  return from_yaml(path.read_text(), expected_cls=expected_cls)

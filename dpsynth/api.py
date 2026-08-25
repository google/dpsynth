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

"""Core abstractions for differentially private mechanisms.

This module defines the ``DPMechanism`` base class, which is the primary
building block for all differentially private algorithms in DP Synth.

Example usage::

  mechanism = dpsynth.discrete_mechanisms.AIMMechanism(pgm_iters=500)

  # Option 1: Calibrate to (epsilon, delta)-DP (tight PLD accounting).
  calibrated = mechanism.calibrate(epsilon=1.0, delta=1e-5)

  # Option 2: Configure with a zCDP budget directly.
  calibrated = mechanism.configure(zcdp_rho=0.5)

  # Run the mechanism.
  result = calibrated(rng, data)
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping
import dataclasses
import functools
import importlib
from typing import Any, TypeVar
import warnings

import dp_accounting
from dpsynth import domain
import yaml

import pathlib
PathType = pathlib.Path | str

SelfConfig = TypeVar('SelfConfig', bound='MechanismConfig')


class CalibratedMechanism(abc.ABC):
  """A privacy-calibrated, runnable differentially private mechanism.

  Produced by ``MechanismConfig.configure()`` / ``.calibrate()``: its natural
  privacy parameter (e.g. Gaussian sigma) is populated and it is ready to run.
  It exposes the exact ``DpEvent`` characterizing its privacy cost and is
  directly callable on data.

  Subclasses must implement:

  - ``dp_event``: return the exact ``DpEvent`` characterizing the mechanism.
  - ``__call__``: run the mechanism on data.
  """

  @property
  @abc.abstractmethod
  def dp_event(self) -> dp_accounting.DpEvent:
    """The DpEvent characterizing the privacy cost of this mechanism."""

  @abc.abstractmethod
  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Runs the mechanism on the given data.

    Subclass signatures vary, but typically accept at least the data to operate
    on and a source of randomness.

    Args:
      *args: Positional arguments (subclass-specific).
      **kwargs: Keyword arguments (subclass-specific).
    """


class MechanismConfig(abc.ABC):
  """A recipe that produces a calibrated, runnable mechanism.

  A config holds the mechanism's structural and hyperparameter fields and knows
  how to turn a privacy budget into a runnable ``CalibratedMechanism``.
  Usage generally follows a three-phase pattern:

  1. **Construct**: Create the config with algorithm-specific parameters
     (e.g., ``AIMConfig(pgm_iters=500)``).
  2. **Calibrate**: Call ``calibrate(epsilon=..., delta=...)`` or
     ``configure(zcdp_rho=...)`` to bind a privacy budget, returning a
     ``CalibratedMechanism`` with the mechanism's natural privacy parameter set.
  3. **Run**: Call the calibrated mechanism on data via ``__call__``.

  **Design: configure vs calibrate.**  The API separates two concerns:

  - ``configure(zcdp_rho, **kwargs)`` is the low-level primitive that each
    config must implement. It maps a zCDP budget to the mechanism's natural
    privacy parameter (e.g., Gaussian sigma) and returns a runnable mechanism.
    This is lightweight — just arithmetic — and produces reasonably tight
    parameter settings for most mechanisms.

  - ``calibrate(epsilon, delta, **kwargs)`` is the high-level entry point
    defined once on the base class. When called with ``(epsilon, delta)``,
    it performs a binary search over zCDP budgets using
    ``dp_accounting.calibrate_dp_mechanism``, calling ``configure`` at each
    candidate and inspecting the resulting ``dp_event`` for tight PLD-based
    accounting. This gives each mechanism the maximum possible budget that
    still satisfies the target (epsilon, delta) guarantee. The (epsilon, delta)
    path is more precise but more expensive than the direct ``zcdp_rho`` path.

  **Why zCDP as the intermediate.**  Calibrating to zCDP rho makes it easy to
  split a privacy budget across a heterogeneous composition of mechanisms:
  simply divide rho additively in any ratio and each share is a valid zCDP
  guarantee.

  **Tight accounting via dp_events.**  Mechanisms may be tighter than their
  zCDP guarantee implies (e.g., GDP mechanisms). The ``calibrate`` binary
  search exploits this: it evaluates each candidate's raw ``dp_event`` rather
  than relying on the zCDP conversion, so the final calibration is as tight
  as the mechanism's own privacy characterization allows.
  """

  _registry: dict[str, type[MechanismConfig]] = {}

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    MechanismConfig._registry[cls.__name__] = cls

  @classmethod
  def get_registered_class(cls, name: str) -> type[MechanismConfig] | None:
    """Returns the registered MechanismConfig subclass for a given name."""
    return cls._registry.get(name)

  def to_dict(self) -> dict[str, Any]:
    """Converts the config into a serializable dictionary."""
    return _config_to_dict(self)

  def to_yaml(self) -> str:
    """Serializes the config into a YAML string."""
    return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

  def to_yaml_file(self, filepath: str | PathType) -> None:
    """Writes the config to a YAML file."""
    with open(filepath, 'w') as f:
      f.write(self.to_yaml())

  @classmethod
  def from_dict(cls: type[SelfConfig], data: Mapping[str, Any]) -> SelfConfig:
    """Instantiates a MechanismConfig from a dictionary."""
    return _config_from_dict(data, expected_cls=cls)

  @classmethod
  def from_yaml(cls: type[SelfConfig], yaml_str: str) -> SelfConfig:
    """Instantiates a MechanismConfig from a YAML string."""
    data = yaml.safe_load(yaml_str)
    if not isinstance(data, dict):
      raise ValueError(f'Expected YAML dictionary, got {type(data).__name__}.')
    return cls.from_dict(data)

  @classmethod
  def from_yaml_file(
      cls: type[SelfConfig], filepath: str | PathType
  ) -> SelfConfig:
    """Reads a MechanismConfig from a YAML file."""
    with open(filepath, 'r') as f:
      return cls.from_yaml(f.read())

  @abc.abstractmethod
  def configure(
      self,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
      max_records_per_user: int = 1,
  ) -> CalibratedMechanism:
    """Returns a calibrated mechanism for the given zCDP budget.

    Converts the zCDP budget into the mechanism's natural privacy parameter
    (e.g., Gaussian sigma) and returns a runnable ``CalibratedMechanism`` with
    that parameter set.

    Most mechanisms are pure zCDP and ignore ``delta``. Mechanisms that
    consume approximate DP budget (e.g., partition selection with Gaussian
    thresholding) should raise ``ValueError`` if ``delta`` is required but
    not provided (i.e., is 0).

    Args:
      zcdp_rho: The zCDP privacy budget (rho).
      delta: Approximate DP delta consumed by the mechanism itself (e.g., for
        thresholding). Defaults to 0 (pure zCDP). Mechanisms that need delta
        should raise if it is 0.
      max_records_per_user: Assumed upper bound on the number of records a
        single user contributes. Values greater than 1 scale the added noise
        (and mechanism sensitivity) to provide user-level rather than
        record-level DP; the privacy accounting is unchanged. This bound is NOT
        enforced -- soundness relies on the caller guaranteeing it via
        preprocessing.

    Returns:
      A calibrated, runnable mechanism.
    """

  def _find_optimal_rho(
      self,
      make_event_fn: Callable[[float], dp_accounting.DpEvent],
      target_epsilon: float,
      target_delta: float,
  ) -> float:
    """Binary-search for the tightest zCDP rho within an (ε, δ) guarantee.

    Tries both RDP and PLD accountants and returns whichever gives the
    highest rho (more budget = better utility). Neither accountant
    universally dominates in tightness.

    Args:
      make_event_fn: Maps a candidate rho to the mechanism's DpEvent.
      target_epsilon: Target epsilon for (epsilon, delta)-DP.
      target_delta: Target delta for (epsilon, delta)-DP.

    Returns:
      The optimal zCDP rho.

    Raises:
      UnsupportedEventError: If no accountant supports the DpEvent.
    """
    rho = float('inf')
    try:
      # This is a heuristic to avoid excessively fine discretization in PLD
      # accounting, which can cause OOM at extremely small target epsilons.
      value_discretization_interval = max(1e-4, 1e-4 / (target_epsilon + 1e-5))
      accountant_fn = functools.partial(
          dp_accounting.pld.PLDAccountant,
          value_discretization_interval=value_discretization_interval,
      )
      rho = dp_accounting.calibrate_dp_mechanism(
          make_fresh_accountant=accountant_fn,
          make_event_from_param=make_event_fn,
          target_epsilon=target_epsilon,
          target_delta=target_delta,
      )
    except (dp_accounting.UnsupportedEventError, NotImplementedError):
      # If PLD accounting is not supported, fall back to RDP accounting.
      pass

    rho2 = dp_accounting.calibrate_dp_mechanism(
        make_fresh_accountant=dp_accounting.rdp.RdpAccountant,
        make_event_from_param=make_event_fn,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
    )

    # RDP can also be better than PLD in some cases due to looseness in the
    # handling of certain DpEvents like the ExponentialMechanismDpEvent.
    return min(rho, rho2)

  def calibrate(
      self,
      *,
      epsilon: float | None = None,
      delta: float | None = None,
      zcdp_rho: float | None = None,
      poisson_sampling_prob: float = 1.0,
      max_records_per_user: int = 1,
      **kwargs: Any,
  ) -> CalibratedMechanism:
    """Calibrate the mechanism to a target (epsilon, delta)-DP guarantee.

    Performs a binary search over zCDP budgets, calling ``configure`` at each
    candidate and inspecting the resulting ``dp_event``. Tries both RDP and
    PLD accounting and picks whichever gives the tightest result.

    .. deprecated::
      Passing ``zcdp_rho`` to ``calibrate`` is deprecated. Use
      ``configure(zcdp_rho=...)`` directly instead.

    Args:
      epsilon: Target epsilon for (epsilon, delta)-DP.
      delta: Target delta for (epsilon, delta)-DP.
      zcdp_rho: Deprecated. Direct zCDP budget. Use ``configure()`` instead.
      poisson_sampling_prob: If specified, calibrate the mechanism assuming the
        input data is subsampleed with the given probability. The actual
        sampling is **NOT** handled internally by the calibrated mechanism.
      max_records_per_user: Assumed upper bound on the number of records a
        single user contributes. Added noise (and mechanism sensitivity) is
        scaled by this factor to provide user-level rather than record-level DP;
        the privacy accounting is unchanged. Soundness relies on the caller
        enforcing this bound.
      **kwargs: Additional mechanism-specific configuration arguments (e.g.
        ``schema`` or ``constraints``).

    Returns:
      A calibrated, runnable mechanism.

    Raises:
      ValueError: If neither (epsilon, delta) nor zcdp_rho is specified, or
        if both are specified simultaneously.
    """
    if zcdp_rho is not None:
      if epsilon is not None or delta is not None:
        raise ValueError(
            'Specify either zcdp_rho or (epsilon, delta), not both.'
        )
      warnings.warn(
          'Passing zcdp_rho to calibrate() is deprecated. Use'
          ' configure(zcdp_rho=...) directly instead.',
          DeprecationWarning,
          stacklevel=2,
      )
      return self.configure(
          zcdp_rho=zcdp_rho,
          max_records_per_user=max_records_per_user,
          **kwargs,
      )

    if epsilon is None or delta is None:
      raise ValueError('Must specify both epsilon and delta, or zcdp_rho.')

    def make_event_fn(rho: float) -> dp_accounting.DpEvent:
      base = self.configure(
          zcdp_rho=rho,
          delta=delta,
          max_records_per_user=max_records_per_user,
          **kwargs,
      ).dp_event
      sampled = dp_accounting.PoissonSampledDpEvent(poisson_sampling_prob, base)
      return base if poisson_sampling_prob == 1.0 else sampled

    optimal_rho = self._find_optimal_rho(
        make_event_fn=make_event_fn,
        target_epsilon=epsilon,
        target_delta=delta,
    )
    return self.configure(
        zcdp_rho=optimal_rho,
        delta=delta,
        max_records_per_user=max_records_per_user,
        **kwargs,
    )


class DPMechanism(MechanismConfig, CalibratedMechanism, abc.ABC):
  """Transitional monolithic base: both a config and a runnable mechanism.

  Historically every mechanism was a single class that was constructed, then
  ``configure``d in place, then run. That design is being split into a
  ``MechanismConfig`` recipe and a ``CalibratedMechanism`` runnable (so a
  calibrated instance can never be un-calibrated and needs no nullable
  privacy-parameter fields or guards). Mechanisms are migrated one layer at a
  time; those not yet split still subclass this monolith, which exposes exactly
  the historical abstract surface (``configure`` + ``dp_event`` + ``__call__``,
  with ``calibrate`` inherited). Remove once every mechanism is split.
  """


def validate_max_records_per_user(value: int) -> None:
  """Raises ValueError if the per-user record bound is not a positive int."""
  if value < 1:
    raise ValueError(f'max_records_per_user must be >= 1, got {value}.')


def _ensure_subclasses_loaded():
  """Loads standard MechanismConfig subclasses to populate registry."""
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
    except (ImportError, AttributeError):
      pass


def _get_foreign_key_relation_cls() -> type[Any] | None:
  """Lazily resolves ForeignKeyRelation class if available."""
  try:
    mod = importlib.import_module('dpsynth.relational.domain')
    return getattr(mod, 'ForeignKeyRelation', None)
  except (ImportError, AttributeError):
    return None


def _config_to_dict(obj: Any) -> Any:
  """Converts an object into a YAML-serializable dictionary structure."""
  if isinstance(obj, MechanismConfig):
    data = {'type': obj.__class__.__name__}
    if dataclasses.is_dataclass(obj):
      for f in dataclasses.fields(obj):
        data[f.name] = _config_to_dict(getattr(obj, f.name))
    return data
  elif isinstance(
      obj,
      (
          domain.CategoricalAttribute,
          domain.NumericalAttribute,
          domain.OpenSetCategoricalAttribute,
          domain.FreeFormTextAttribute,
      ),
  ):
    return domain.attribute_to_dict(obj)
  elif isinstance(obj, domain.Schema):
    return obj.to_dict()
  elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
    return obj.to_dict()
  elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    data = {}
    for f in dataclasses.fields(obj):
      data[f.name] = _config_to_dict(getattr(obj, f.name))
    data['type'] = obj.__class__.__name__
    return data
  elif isinstance(obj, Mapping):
    return {str(k): _config_to_dict(v) for k, v in obj.items()}
  elif isinstance(obj, (list, tuple)):
    return [_config_to_dict(item) for item in obj]
  else:
    return obj


def _instantiate_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
  """Instantiates a dataclass, converting nested fields appropriately."""
  kwargs = {}
  for field in dataclasses.fields(cls):
    if field.name not in data:
      continue
    val = data[field.name]
    if val is None:
      kwargs[field.name] = None
      continue

    if field.name in ('domains', 'domain', 'schema'):
      if isinstance(val, dict):
        if 'attributes' in val:
          kwargs[field.name] = domain.Schema.from_dict(val)
        else:
          first_val = next(iter(val.values()), None)
          if isinstance(first_val, dict) and any(
              k in first_val
              for k in (
                  'min_value',
                  'possible_values',
                  'default_value',
                  'max_tokens',
                  'type',
              )
          ):
            kwargs[field.name] = domain.Schema.from_dict(val)
          elif isinstance(first_val, dict):
            kwargs[field.name] = {
                t: domain.Schema.from_dict(c) for t, c in val.items()
            }
          else:
            kwargs[field.name] = val
      else:
        kwargs[field.name] = val
    elif field.name in ('constraints', 'cross_attribute_constraints'):
      if isinstance(val, list):
        constraints_mod = importlib.import_module('dpsynth.constraints')
        c_list = [
            constraints_mod.Constraint.from_dict(c)
            if isinstance(c, dict)
            else c
            for c in val
        ]
        if isinstance(field.default, tuple):
          kwargs[field.name] = tuple(c_list)
        else:
          kwargs[field.name] = c_list
      else:
        kwargs[field.name] = val
    elif field.name in ('discrete_mechanism', 'mechanism'):
      kwargs[field.name] = _config_from_dict(val, expected_cls=MechanismConfig)
    elif field.name == 'initializers':
      if isinstance(val, dict):
        res = {}
        for k, v in val.items():
          if isinstance(v, dict):
            if 'type' in v or any(
                f in v for f in ('target_attribute_name', 'num_bins')
            ):
              res[k] = _config_from_dict(v, expected_cls=MechanismConfig)
            else:
              res[k] = {
                  col: _config_from_dict(cfg, expected_cls=MechanismConfig)
                  for col, cfg in v.items()
              }
          else:
            res[k] = v
        kwargs[field.name] = res
      else:
        kwargs[field.name] = val
    elif field.name in ('foreign_keys', 'relations'):
      fkr_cls = _get_foreign_key_relation_cls()
      if isinstance(val, (list, tuple)) and fkr_cls is not None:
        fk_list = []
        for item in val:
          if isinstance(item, dict):
            item_dict = {k: v for k, v in item.items() if k != 'type'}
            fk_list.append(fkr_cls(**item_dict))
          else:
            fk_list.append(item)
        if isinstance(field.default, tuple):
          kwargs[field.name] = tuple(fk_list)
        else:
          kwargs[field.name] = fk_list
      else:
        kwargs[field.name] = val
    elif field.name == 'workload':
      if isinstance(val, list):
        kwargs[field.name] = [
            tuple(c) if isinstance(c, list) else c for c in val
        ]
      elif isinstance(val, dict):
        kwargs[field.name] = {
            tuple(k) if isinstance(k, list) else k: v for k, v in val.items()
        }
      else:
        kwargs[field.name] = val
    elif field.name == 'prespecified_marginal_queries':
      if isinstance(val, list):
        kwargs[field.name] = [
            tuple(q) if isinstance(q, list) else q for q in val
        ]
      else:
        kwargs[field.name] = val
    elif field.name == 'attribute' and isinstance(val, dict):
      kwargs[field.name] = domain.attribute_from_dict(val)
    elif isinstance(val, dict) and 'type' in val:
      kwargs[field.name] = _config_from_dict(val)
    elif isinstance(field.default, tuple) and isinstance(val, list):
      kwargs[field.name] = tuple(val)
    else:
      kwargs[field.name] = val
  return cls(**kwargs)


def _config_from_dict(
    data: Mapping[str, Any], expected_cls: type[Any] | None = None
) -> Any:
  """Reconstructs a MechanismConfig or nested object from a dictionary."""
  if not isinstance(data, dict):
    return data

  _ensure_subclasses_loaded()

  type_name = data.get('type')
  if type_name:
    data_without_type = {k: v for k, v in data.items() if k != 'type'}
    target_cls = MechanismConfig.get_registered_class(type_name)
    if target_cls is not None:
      return _instantiate_dataclass(target_cls, data_without_type)
    elif type_name in (
        'CategoricalAttribute',
        'NumericalAttribute',
        'OpenSetCategoricalAttribute',
        'FreeFormTextAttribute',
    ):
      return domain.attribute_from_dict(data)
    elif type_name == 'ForeignKeyRelation':
      fkr_cls = _get_foreign_key_relation_cls()
      if fkr_cls is not None:
        return fkr_cls(**data_without_type)
      raise ValueError(f"Unknown type in YAML: '{type_name}'")
    else:
      raise ValueError(f"Unknown type in YAML: '{type_name}'")
  elif expected_cls is not None and expected_cls is not MechanismConfig:
    if dataclasses.is_dataclass(expected_cls):
      return _instantiate_dataclass(expected_cls, dict(data))
    return expected_cls(**data)
  else:
    raise ValueError(
        "Missing 'type' field in YAML dictionary to identify MechanismConfig"
        ' class.'
    )


def to_yaml(config: MechanismConfig) -> str:
  """Serializes a MechanismConfig into a YAML string.

  Args:
    config: The MechanismConfig to serialize.

  Returns:
    A YAML string representation of the config.
  """
  return config.to_yaml()


def to_yaml_file(config: MechanismConfig, filepath: str | PathType) -> None:
  """Writes a MechanismConfig to a YAML file.

  Args:
    config: The MechanismConfig to serialize.
    filepath: Destination file path.
  """
  config.to_yaml_file(filepath)


def from_yaml(yaml_str: str) -> MechanismConfig:
  """Loads a MechanismConfig from a YAML string.

  Args:
    yaml_str: YAML string encoding a MechanismConfig.

  Returns:
    The reconstructed MechanismConfig instance.
  """
  return MechanismConfig.from_yaml(yaml_str)


def from_yaml_file(filepath: str | PathType) -> MechanismConfig:
  """Loads a MechanismConfig from a YAML file.

  Args:
    filepath: Path to the YAML file.

  Returns:
    The reconstructed MechanismConfig instance.
  """
  return MechanismConfig.from_yaml_file(filepath)

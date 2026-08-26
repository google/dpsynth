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

"""Public API for DPSynth."""

# pylint: disable=g-importing-member
__version__ = '0.4.0'
from dpsynth import api
from dpsynth import constraints
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth import relational
from dpsynth import serialization
from dpsynth.data_generation_v3 import TabularConfig
from dpsynth.data_generation_v3 import TabularMechanism
from dpsynth.data_generation_v3 import TabularSynthesizer
from dpsynth.discrete_mechanisms.discrete import DiscreteConfig
from dpsynth.discrete_mechanisms.discrete import DiscreteMechanism
from dpsynth.domain import CategoricalAttribute
from dpsynth.domain import FreeFormTextAttribute
from dpsynth.domain import NumericalAttribute
from dpsynth.domain import OpenSetCategoricalAttribute
from dpsynth.domain import Schema
from dpsynth.serialization import from_yaml
from dpsynth.serialization import from_yaml_file
from dpsynth.serialization import to_yaml
from dpsynth.serialization import to_yaml_file

ForeignKeyRelation = relational.ForeignKeyRelation
MultiDataGenerationResult = relational.MultiDataGenerationResult
MultiTableConfig = relational.MultiTableConfig
MultiTableMechanism = relational.MultiTableMechanism

__all__ = [
    'CategoricalAttribute',
    'ForeignKeyRelation',
    'MultiDataGenerationResult',
    'MultiTableConfig',
    'MultiTableMechanism',
    'NumericalAttribute',
    'OpenSetCategoricalAttribute',
    'Schema',
    'TabularConfig',
    'TabularMechanism',
    'TabularSynthesizer',
    'api',
    'discrete_mechanisms',
    'domain',
    'from_yaml',
    'from_yaml_file',
    'relational',
    'serialization',
    'to_yaml',
    'to_yaml_file',
]

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

"""Implementations of mechanisms that operate over discrete data."""

# pylint: disable=g-importing-member

from dpsynth.discrete_mechanisms.aim import AIMMechanism
from dpsynth.discrete_mechanisms.aim_gdp import AIMGDPMechanism
from dpsynth.discrete_mechanisms.common import DiscreteMechanismResult
from dpsynth.discrete_mechanisms.direct import DirectMechanism
from dpsynth.discrete_mechanisms.independent import IndependentMechanism
from dpsynth.discrete_mechanisms.mst import MSTMechanism
from dpsynth.discrete_mechanisms.swift import SWIFTMechanism
from dpsynth.local_mode.primitives import DPMechanism as DiscreteMechanism

# Backwards-compatible aliases.
AIMConfig = AIMMechanism
AIMGDPConfig = AIMGDPMechanism
DirectConfig = DirectMechanism
IndependentConfig = IndependentMechanism
MSTConfig = MSTMechanism
SWIFTConfig = SWIFTMechanism

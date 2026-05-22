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

"""Common types for the pipeline transformations."""

from collections.abc import Iterable
import enum
from typing import TypeAlias, TypeVar

import apache_beam as beam
import numpy as np

T = TypeVar('T')
Collection: TypeAlias = Iterable[T] | beam.PCollection[T]


Clique: TypeAlias = tuple[int, ...]
Record: TypeAlias = tuple[int, ...]
Marginal: TypeAlias = tuple[Clique, np.ndarray]
Cliques: TypeAlias = list[Clique]


class DataFormat(enum.Enum):
  """Data format to use for data generation."""

  CSV = 'csv'
  TFRECORD = 'tfrecord'

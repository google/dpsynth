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

"""Types and configurations for tabular evaluation."""

from collections.abc import Iterable
import enum
from typing import Any, TypeAlias, TypeVar

import apache_beam as beam


T = TypeVar("T")
Collection: TypeAlias = Iterable[T] | beam.PCollection[T]
Record: TypeAlias = tuple[Any, ...]


# We can extend the pb2 enum with helper methods.
class DataType(enum.Enum):
  """Wrapper around the enum."""

  DATA_TYPE_UNSPECIFIED = 0
  INT_CATEGORICAL = 1
  STRING = 2
  BOOLEAN = 3

  def is_categorical(self) -> bool:
    return self in (DataType.INT_CATEGORICAL, DataType.STRING, DataType.BOOLEAN)

  def is_numerical(self) -> bool:
    return False

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, DataType):
      return self.value == other.value
    return self.value == other

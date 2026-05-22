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

"""Dataset descriptor for CSV data."""

import numbers
from typing import Any

from dpsynth.dataset_descriptors import dataset_descriptor
import pandas as pd

DataType = dataset_descriptor.DataType


class CSVConverter(dataset_descriptor.DataRecordConverter):
  """Converts between dataframe rows and tuples."""

  def __init__(self, attributes: dict[str, DataType]):
    self._attributes = attributes

  @property
  def attributes(self) -> dict[str, DataType]:
    return self._attributes

  def _check_value_type(
      self, value: Any, expected_type: DataType, attribute: str
  ):
    if expected_type == DataType.INT and not isinstance(
        value, numbers.Integral
    ):
      raise ValueError(
          f"Expected type int for attribute {attribute}, got {type(value)}"
      )
    if expected_type == DataType.STR and not isinstance(value, str):
      raise ValueError(
          f"Expected type str for attribute {attribute}, got {type(value)}"
      )
    if expected_type == DataType.BOOL and not isinstance(value, bool):
      raise ValueError(
          f"Expected type bool for attribute {attribute}, got {type(value)}"
      )
    if expected_type == DataType.FLOAT and not isinstance(value, numbers.Real):
      raise ValueError(
          f"Expected type float for attribute {attribute}, got {type(value)}"
      )

  def to_tuple(self, df_row: tuple[Any, pd.Series]) -> tuple[Any, ...]:
    row_data = df_row[1]
    for attribute, expected_type in self._attributes.items():
      value = getattr(row_data, attribute)
      self._check_value_type(value, expected_type, attribute)
    return tuple(
        getattr(row_data, attribute) for attribute in self._attributes.keys()
    )

  def from_tuple(self, tuple_row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple_row


def _deduce_column_data_types(
    df: pd.DataFrame, field_names: list[str]
) -> dict[str, DataType]:
  """Deduces the data type of each column in a DataFrame.

  Args:
      df: The input DataFrame.
      field_names: The names of the columns to deduce the data type for.

  Returns:
      A dict mapping column names to their data types.
  """
  attributes = {}
  for column in field_names:
    if column not in df.columns:
      raise ValueError(f"Column '{column}' not found in DataFrame.")
    dtype = df.dtypes[column]
    if pd.api.types.is_integer_dtype(dtype):
      attributes[column] = DataType.INT
    elif pd.api.types.is_string_dtype(dtype):
      attributes[column] = DataType.STR
    elif pd.api.types.is_bool_dtype(dtype):
      attributes[column] = DataType.BOOL
    elif pd.api.types.is_float_dtype(dtype):
      attributes[column] = DataType.FLOAT
    else:
      attributes[column] = DataType.UNSUPPORTED
  return attributes


def read_csv_sample(path: str) -> pd.DataFrame:
  """Loads first 1000 rows of data from a CSV file."""
  return pd.read_csv(path, nrows=1000)


def get_dataset_descriptor_for_csv(
    dataframe: pd.DataFrame,
    field_names: list[str] | None = None,
) -> dataset_descriptor.DatasetDescriptor:
  """Creates a DatasetDescriptor for a CSV file.

  Args:
      dataframe: The input pandas DataFrame loaded from the first 1000 rows of a
        CSV file.
      field_names: The names of the columns to include in the descriptor. If
        None, all columns are included.

  Returns:
      A DatasetDescriptor object explaining the CSV file with supported
      attributes.
  """
  if field_names is None:
    field_names = dataframe.columns
  attributes = _deduce_column_data_types(dataframe, field_names)
  dataset_attributes = []
  for attribute in attributes:
    if attributes[attribute] == DataType.UNSUPPORTED:
      continue
    dataset_attributes.append(
        dataset_descriptor.AttributeDescriptor(
            name=attribute, data_type=attributes[attribute]
        )
    )
  return dataset_descriptor.DatasetDescriptor(
      attributes=dataset_attributes,
      data_record_converter=CSVConverter(attributes),
  )

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

"""Derive the domain from a given dataset and write it out to storage."""

import os

from absl import app
from absl import flags
from absl import logging
from dpsynth import domain
from dpsynth.bin import _read_csv_args
import numpy as np
import pandas as pd

import pathlib
PathType = pathlib.Path

_DATASET_PATH = flags.DEFINE_string(
    'dataset_path',
    'adult.csv',
    'Path to the dataset to derive the domain from.',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to the output directory to write the domain to.',
)

_CSV_READ_ARGS_FIELD_SEPARATOR = flags.DEFINE_string(
    'csv_read_args_field_separator',
    None,
    'Field separator for reading CSV files.',
)
_CSV_READ_ARGS_COLUMN_NAMES = flags.DEFINE_list(
    'csv_read_args_column_names',
    None,
    'Column names for reading CSV files.',
)
_CSV_READ_ARGS_COLUMN_COUNT = flags.DEFINE_integer(
    'csv_read_args_column_count',
    None,
    'Column count for reading CSV files.',
)


_NUMERICAL_SENTINEL_VALUE = flags.DEFINE_integer(
    'numerical_sentinel_value',
    None,
    'Sentinel value to use for numerical columns.',
)


def _create_numerical_attribute(
    df_col: pd.Series,
    dtype_str: str,
    numerical_sentinel_value: int | None = None,
) -> domain.NumericalAttribute | domain.CategoricalAttribute:
  """Creates a numerical attribute from a pandas Series."""
  clip_to_range = True
  if numerical_sentinel_value is not None and any(
      value == numerical_sentinel_value for value in df_col.values
  ):
    # Replace the sentinel value with NaN so that it is not used in the range
    # computation.
    df_col = df_col.replace(numerical_sentinel_value, np.nan)
    clip_to_range = False

  max_value = df_col.max()
  min_value = df_col.min()

  # If all values are NaN, return a categorical attribute with a single
  # sentinel value or None. This typically happens when the column is empty.
  if np.isnan(min_value):
    if not np.isnan(max_value):
      raise ValueError(
          'max_value is not NaN but min_value can be NaN only if all values are'
          ' NaN. This is unexpected.'
      )

    return domain.CategoricalAttribute(
        possible_values=[
            numerical_sentinel_value if numerical_sentinel_value else None
        ]
    )

  # Check that the sentinel value is not in the range [min_value, max_value].
  if (
      numerical_sentinel_value is not None
      and numerical_sentinel_value >= min_value
      and numerical_sentinel_value <= max_value
  ):
    raise ValueError(
        f'Sentinel value {numerical_sentinel_value} should be outside the range'
        f' [{min_value}, {max_value}]'
    )

  # If the min and max values are the same, return a categorical attribute with
  # a single value.
  if min_value == max_value:
    return domain.CategoricalAttribute(possible_values=[min_value])

  return domain.NumericalAttribute(
      min_value=min_value,
      max_value=max_value,
      clip_to_range=clip_to_range,
      dtype=dtype_str,
  )


def derive_domain_from_data(
    df: pd.DataFrame,
    numerical_sentinel_value: int | None = None,
) -> dict[str, domain.AttributeType]:
  """Derive the domain from a given dataset."""
  result = {}
  for col in df.columns:
    logging.info('Deriving domain for column: %s', col)
    match df[col].dtype:
      case 'object' | 'str' | 'string' | 'category':
        result[col] = domain.CategoricalAttribute(
            possible_values=sorted(
                df[col].unique(),
                key=lambda x: (isinstance(x, str), x),  # sort ints before strs.
            )
        )
      case 'int' | 'int64' | 'int32':
        result[col] = _create_numerical_attribute(
            df[col], 'int', numerical_sentinel_value
        )
      case 'float' | 'float64' | 'float32':
        result[col] = _create_numerical_attribute(
            df[col], 'float', numerical_sentinel_value
        )
      case _:
        raise ValueError(f'Unsupported dtype: {df[col].dtype}')
  return result


def _get_yaml_filename(dataset_path: PathType) -> str:
  return os.path.basename(dataset_path) + '_domain.yaml'


def main(_) -> None:
  csv_read_args = _read_csv_args.ReadCsvArgs(
      field_separator=_CSV_READ_ARGS_FIELD_SEPARATOR.value,
      column_names=_CSV_READ_ARGS_COLUMN_NAMES.value,
      column_count=_CSV_READ_ARGS_COLUMN_COUNT.value,
  )
  read_csv_kwargs = csv_read_args.to_read_csv_kwargs()

  # If output_dir is not set, use the parent directory of the dataset path.
  output_dir = _OUTPUT_DIR.value
  if not output_dir:
    output_dir = _DATASET_PATH.value.parent

  dataset_path = pathlib.Path(_DATASET_PATH.value)
  if output_dir:
    output_dir_path = pathlib.Path(output_dir)
  else:
    output_dir_path = dataset_path.parent
  yaml_path = output_dir_path / _get_yaml_filename(dataset_path)

  df = pd.read_csv(
      filepath_or_buffer=str(_DATASET_PATH.value), **read_csv_kwargs
  )
  dom = derive_domain_from_data(df, _NUMERICAL_SENTINEL_VALUE.value)
  logging.info('Writing domain to %s', yaml_path)
  domain.to_yaml_file(dom, yaml_path)


if __name__ == '__main__':
  app.run(main)

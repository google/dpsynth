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

import collections

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth.dataset_descriptors import csv_descriptor
from dpsynth.dataset_descriptors import dataset_descriptor
import pandas as pd

OrderedDict = collections.OrderedDict


class CsvDescriptorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="all_supported_types",
          df=pd.DataFrame({
              "int_col": [1, 2, 3],
              "float_col": [1.0, 2.0, 3.0],
              "str_col": ["a", "b", "c"],
              "bool_col": [True, False, True],
          }),
          expected_attributes=OrderedDict({
              "int_col": dataset_descriptor.DataType.INT,
              "float_col": dataset_descriptor.DataType.FLOAT,
              "str_col": dataset_descriptor.DataType.STR,
              "bool_col": dataset_descriptor.DataType.BOOL,
          }),
      ),
      dict(
          testcase_name="mixed_types",
          df=pd.DataFrame({
              "mixed_col": [1, 2.0, "3"],
          }),
          expected_attributes=OrderedDict({
              "mixed_col": dataset_descriptor.DataType.STR,
          }),
      ),
      dict(
          testcase_name="empty_column",
          df=pd.DataFrame({
              "empty_col": [],
          }),
          expected_attributes=OrderedDict({
              "empty_col": dataset_descriptor.DataType.FLOAT,
          }),
      ),
      dict(
          testcase_name="single_field_name",
          df=pd.DataFrame({
              "int_col": [1, 2, 3],
              "float_col": [1.0, 2.0, 3.0],
              "str_col": ["a", "b", "c"],
              "bool_col": [True, False, True],
          }),
          expected_attributes=OrderedDict(
              {"str_col": dataset_descriptor.DataType.STR}
          ),
          field_names=["str_col"],
      ),
  )
  def test_deduce_column_data_types(
      self, df, expected_attributes, field_names=None
  ):
    if field_names is None:
      field_names = df.columns
    attributes = csv_descriptor._deduce_column_data_types(df, field_names)
    self.assertEqual(attributes, expected_attributes)

  def test_deduce_column_data_types_with_nonexistent_field_name(self):
    with self.assertRaisesRegex(
        ValueError, "Column 'str_col' not found in DataFrame."
    ):
      csv_descriptor._deduce_column_data_types(
          df=pd.DataFrame({
              "int_col": [1, 2, 3],
          }),
          field_names=["str_col"],
      )

  def test_to_tuple(self):
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
    })
    descriptor = csv_descriptor.get_dataset_descriptor_for_csv(df)
    input_df_row = list(df.iterrows())[0]
    self.assertEqual(
        descriptor.data_record_converter.to_tuple(input_df_row), (1, "a", True)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_str_type",
          df_invalid=pd.DataFrame({
              "int_col": [1],
              "str_col": [2],
              "bool_col": [True],
              "float_col": [1.0],
          }),
          expected_error="Expected type str for attribute str_col",
      ),
      dict(
          testcase_name="invalid_int_type",
          df_invalid=pd.DataFrame({
              "int_col": ["a"],
              "str_col": ["b"],
              "bool_col": [True],
              "float_col": [1.0],
          }),
          expected_error="Expected type int for attribute int_col",
      ),
      dict(
          testcase_name="invalid_bool_type",
          df_invalid=pd.DataFrame({
              "int_col": [1],
              "str_col": ["a"],
              "bool_col": [3],
              "float_col": [1.0],
          }),
          expected_error="Expected type bool for attribute bool_col",
      ),
      dict(
          testcase_name="invalid_float_type",
          df_invalid=pd.DataFrame({
              "int_col": [1],
              "str_col": ["a"],
              "bool_col": [True],
              "float_col": ["b"],
          }),
          expected_error="Expected type float for attribute float_col",
      ),
  )
  def test_to_tuple_with_invalid_types(self, df_invalid, expected_error):
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
        "float_col": [1.0, 2.0, 3.0],
    })
    descriptor = csv_descriptor.get_dataset_descriptor_for_csv(df)
    input_df_row = list(df_invalid.iterrows())[0]
    with self.assertRaisesRegex(ValueError, expected_error):
      descriptor.data_record_converter.to_tuple(input_df_row)

  def test_from_tuple(self):
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
    })
    descriptor = csv_descriptor.get_dataset_descriptor_for_csv(df)
    input_tuple = (1, "a", True)
    self.assertEqual(
        descriptor.data_record_converter.from_tuple(input_tuple), input_tuple
    )

  def test_csv_converter_attributes(self):
    """Tests that the CSVConverter correctly stores the provided attributes."""
    attributes = collections.OrderedDict([
        ("int_col", dataset_descriptor.DataType.INT),
        ("str_col", dataset_descriptor.DataType.STR),
    ])
    converter = csv_descriptor.CSVConverter(attributes)
    self.assertEqual(converter.attributes, attributes)


if __name__ == "__main__":
  absltest.main()

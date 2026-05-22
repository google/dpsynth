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

from absl.testing import absltest
from dpsynth.dataset_descriptors import creating_data_recorder_converter
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import types
import pandas as pd
import tensorflow as tf


class MockDataRecordConverter(dataset_descriptor.DataRecordConverter):

  def to_tuple(self, record):
    return tuple(record)

  def from_tuple(self, record, not_used_proto_object=None):
    return record


class CreatingDataRecorderConverterTest(absltest.TestCase):

  def test_create_csv_data_record_converter(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="test_long",
            data_type=dataset_descriptor.DataType.INT,
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )

    csv_converter = (
        creating_data_recorder_converter.create_data_record_converter(
            desc, types.DataFormat.CSV
        )
    )
    self.assertIsNotNone(csv_converter)

    series = pd.Series({"test_long": 42})
    df_row = (0, series)
    tup = csv_converter.to_tuple(df_row)
    self.assertEqual(tup, (42,))
    self.assertEqual(csv_converter.from_tuple((42,)), (42,))

  def test_create_tfrecord_data_record_converter(self):
    attributes = [
        dataset_descriptor.AttributeDescriptor(
            name="test_long",
            data_type=dataset_descriptor.DataType.INT,
        ),
    ]
    desc = dataset_descriptor.DatasetDescriptor(
        attributes=attributes, data_record_converter=MockDataRecordConverter()
    )

    tfrecord_converter = (
        creating_data_recorder_converter.create_data_record_converter(
            desc, types.DataFormat.TFRECORD
        )
    )
    self.assertIsNotNone(tfrecord_converter)

    example = tf.train.Example()
    example.features.feature["test_long"].int64_list.value.append(42)
    tup = tfrecord_converter.to_tuple(example)
    self.assertEqual(tup, (42,))

    example_from_tup = tfrecord_converter.from_tuple((42,))
    self.assertEqual(
        example_from_tup.features.feature["test_long"].int64_list.value[0], 42
    )


if __name__ == "__main__":
  absltest.main()

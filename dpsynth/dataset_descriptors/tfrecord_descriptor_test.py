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
import os
import pickle

from absl.testing import absltest
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.dataset_descriptors import tfrecord_descriptor
import tensorflow as tf


class TfrecordDescriptorTest(absltest.TestCase):

  def test_get_dataset_descriptor_for_tfrecord(self):
    sample_records = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string"])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[10])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[2])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string2"])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[20])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[3])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[30])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string3"])
                    ),
                }
            )
        ),
    ]

    descriptor = tfrecord_descriptor.get_dataset_descriptor_for_tfrecord(
        sample_records
    )

    self.assertCountEqual(
        descriptor.attributes,
        [
            dataset_descriptor.AttributeDescriptor(
                name="int_feature", data_type=dataset_descriptor.DataType.INT
            ),
            dataset_descriptor.AttributeDescriptor(
                name="int_feature2", data_type=dataset_descriptor.DataType.INT
            ),
            dataset_descriptor.AttributeDescriptor(
                name="string_feature", data_type=dataset_descriptor.DataType.STR
            ),
        ],
    )

  def test_mismatched_features_in_sample_records(self):
    sample_records = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string"])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string2"])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[20])
                    ),
                }
            )
        ),
    ]

    with self.assertRaises(ValueError):
      tfrecord_descriptor.get_dataset_descriptor_for_tfrecord(sample_records)

  def test_additional_features_in_sample_records(self):
    sample_records = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[2])
                    ),
                }
            )
        ),
    ]

    descriptor = tfrecord_descriptor.get_dataset_descriptor_for_tfrecord(
        sample_records
    )
    self.assertEqual(
        descriptor.attributes,
        [
            dataset_descriptor.AttributeDescriptor(
                name="int_feature", data_type=dataset_descriptor.DataType.INT
            ),
        ],
    )

  def test_get_dataset_descriptor_with_attribute_selection(self):
    sample_records = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string"])
                    ),
                }
            )
        ),
    ]

    descriptor = tfrecord_descriptor.get_dataset_descriptor_for_tfrecord(
        sample_records, attributes=["int_feature"]
    )

    self.assertEqual(
        descriptor.attributes,
        [
            dataset_descriptor.AttributeDescriptor(
                name="int_feature", data_type=dataset_descriptor.DataType.INT
            ),
        ],
    )

  def test_mismatched_feature_types_in_sample_records(self):
    sample_records = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "feature_1": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "feature_1": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string2"])
                    ),
                    "float_feature": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[1.0])
                    ),
                }
            )
        ),
    ]

    with self.assertRaisesRegex(ValueError, "Record has feature"):
      tfrecord_descriptor.get_dataset_descriptor_for_tfrecord(sample_records)

  def test_invalid_feature_values(self):
    sample_records = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "feature_1": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1, 2])
                    ),
                }
            )
        ),
    ]

    with self.assertRaisesRegex(
        ValueError, "Record has feature feature_1 with 2 values."
    ):
      tfrecord_descriptor.get_dataset_descriptor_for_tfrecord(sample_records)

  def test_no_sample_records(self):
    with self.assertRaisesRegex(ValueError, "No sample records provided."):
      tfrecord_descriptor.get_dataset_descriptor_for_tfrecord([])

  def test_converts_to_from_tuple(self):
    descriptor = tfrecord_descriptor.get_dataset_descriptor_for_tfrecord(
        [
            tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "int_feature": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[1])
                        ),
                        "string_feature": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[b"s"])
                        ),
                        "float_feature": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[1.0])
                        ),
                    }
                )
            ),
        ],
        attributes=["int_feature", "string_feature", "float_feature"],
    )

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "int_feature": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[3])
                ),
                "string_feature": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[b"v"])
                ),
                "float_feature": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[1.5])
                ),
            }
        )
    )

    tuple_record = descriptor.data_record_converter.to_tuple(example)
    # Order is not guaranteed, because the order in tf record is not guaranteed.
    self.assertEqual(set(tuple_record), set([3, "v", 1.5]))

    example_from_tuple = descriptor.data_record_converter.from_tuple(
        tuple_record
    )

    self.assertEqual(example, example_from_tuple)

  def test_read_tfrecords_sample(self):
    records = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string"])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[10])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[2])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string2"])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[20])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[3])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[30])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string3"])
                    ),
                }
            )
        ),
    ]
    output_file = self.create_tempfile()
    with tf.io.TFRecordWriter(output_file.full_path) as writer:
      for record in records:
        writer.write(record.SerializeToString())

    sample_records = tfrecord_descriptor.read_tfrecords_sample(
        output_file.full_path,
        sample_size=2,
    )

    self.assertEqual(sample_records, records[:2])

  def test_read_tfrecords_sample_with_sharded_file(self):
    records = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string"])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[10])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[2])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string2"])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[20])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[3])
                    ),
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[30])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string3"])
                    ),
                }
            )
        ),
    ]

    tempdir = self.create_tempdir()
    sharded_file_spec = "sharded_file@2"
    sharded_file_glob = "sharded_file*"
    shards = self._generate_sharded_filenames(sharded_file_spec)
    for shard in shards:
      output_file = tempdir.create_file(shard)
      with tf.io.TFRecordWriter(output_file.full_path) as writer:
        for record in records:
          writer.write(record.SerializeToString())

    sample_records = tfrecord_descriptor.read_tfrecords_sample(
        os.path.join(tempdir.full_path, sharded_file_glob),
        sample_size=2,
    )

    self.assertEqual(sample_records, records[:2])

  def test_serialization_deserialization(self):
    descriptor = tfrecord_descriptor.get_dataset_descriptor_for_tfrecord([
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int_feature2": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[5])
                    ),
                    "int_feature3": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[15])
                    ),
                    "int_feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[3])
                    ),
                    "string_feature": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"string"])
                    ),
                }
            )
        )
    ])

    serialized = pickle.dumps(descriptor)
    deserialized = pickle.loads(serialized)

    self.assertEqual(descriptor.attributes, deserialized.attributes)

  def test_tfrecord_converter_attributes(self):
    """Tests that TFRecordConverter correctly stores the attributes."""
    attributes = collections.OrderedDict(
        [("int_feature", dataset_descriptor.DataType.INT)]
    )
    converter = tfrecord_descriptor.TFRecordConverter(attributes)
    self.assertEqual(converter.attributes, attributes)

  def _generate_sharded_filenames(self, spec: str) -> list[str]:
    name, count = spec.split("@")
    count = int(count)
    return [f"{name}-{i:05d}-of-{count:05d}" for i in range(count)]


if __name__ == "__main__":
  absltest.main()

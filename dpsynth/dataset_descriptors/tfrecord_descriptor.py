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

"""DatasetDescriptor for TFRecords."""

from typing import Any

from dpsynth.dataset_descriptors import dataset_descriptor
import tensorflow as tf

import glob
glob_func = glob.glob


class TFRecordConverter(dataset_descriptor.DataRecordConverter):
  """Converts between TFRecord records and tuples.

  As of now, the converter only supports integer attributes.
  """

  def __init__(self, attributes: dict[str, dataset_descriptor.DataType]):
    self._attributes = attributes

  @property
  def attributes(self) -> dict[str, dataset_descriptor.DataType]:
    return self._attributes

  def to_tuple(self, record: tf.train.Example) -> tuple[Any, ...]:
    """Converts a TFRecord record to a tuple.

    Unsupported attributes are ignored.

    Args:
      record: A TFRecord record.

    Returns:
      A tuple of values of the attributes in the record.
    """
    return tuple(
        self._get_value(attribute, record)
        for attribute in self._attributes.keys()
    )

  def from_tuple(
      self, record: tuple[Any, ...], proto_object: Any | None = None
  ) -> tf.train.Example:
    """Converts a tuple of values to a TFRecord record.

    Unsupported attributes are ignored.

    Args:
      record: A tuple of values of the attributes.
      proto_object: not used

    Returns:
      A tensorflow.Example record.
    """
    example = tf.train.Example()
    for attribute, value in zip(self._attributes.keys(), record):
      self._set_value(attribute, example, value)
    return example

  def _get_value(self, attribute: str, record: tf.train.Example) -> Any:
    if attribute not in record.features.feature:
      return None
    if self._attributes[attribute] == dataset_descriptor.DataType.INT:
      values = record.features.feature[attribute].int64_list.value
      return values[0] if values else None

    if self._attributes[attribute] == dataset_descriptor.DataType.STR:
      values = record.features.feature[attribute].bytes_list.value
      return values[0].decode("utf-8") if values else None

    if self._attributes[attribute] == dataset_descriptor.DataType.FLOAT:
      values = record.features.feature[attribute].float_list.value
      return values[0] if values else None
    return None

  def _set_value(
      self, attribute: str, tf_example: tf.train.Example, value: Any
  ) -> None:
    if self._attributes[attribute] in [
        dataset_descriptor.DataType.INT,
        dataset_descriptor.DataType.ENUM,
        dataset_descriptor.DataType.BOOL,
    ]:
      if value is None:
        # out of domain value, i.e. it was not chosen by partition selection
        # for this dataset.
        value = -1
      tf_example.features.feature[attribute].int64_list.value.append(value)
    elif self._attributes[attribute] == dataset_descriptor.DataType.STR:
      if value is None:
        # out of domain value, i.e. it was not chosen by partition selection
        # for this dataset.
        value = ""
      tf_example.features.feature[attribute].bytes_list.value.append(
          value.encode("utf-8")
      )
    elif self._attributes[attribute] == dataset_descriptor.DataType.FLOAT:
      # for now.
      if value is not None:
        tf_example.features.feature[attribute].float_list.value.append(value)


def read_tfrecords_sample(
    path: str, sample_size: int = 1000
) -> list[tf.train.Example]:
  """Reads a sample of TFRecords from a file."""
  dataset = tf.data.TFRecordDataset(glob_func(path))  # pyrefly: ignore[bad-instantiation]
  return [
      tf.train.Example.FromString(record.numpy())
      for record in dataset.take(sample_size)
  ]


def get_dataset_descriptor_for_tfrecord(
    sample_records: list[tf.train.Example],
    *,
    attributes: list[str] | None = None,
) -> dataset_descriptor.DatasetDescriptor:
  """Returns a DatasetDescriptor for a TFRecord dataset.

  Args:
    sample_records: A representative sample of TFRecords dataset to infer the
      attributes and their types. This could be a subset of the full dataset for
      performance reasons. These records are parsed to infer the attributes and
      their types. It is assumed that all records have the same set of
      attributes with same data types, hence the operation does not require to
      be Differentially Private.
    attributes: A list of attributes to keep from the TFRecord dataset. If None,
      all attributes are kept.

  Raises:
    ValueError: If no sample records are provided or provided records do not
      have the same set of attributes or same data types.

  Returns:
    A DatasetDescriptor object explaining the tfrecord dataset with supported
      attributes.
  """
  if not sample_records:
    raise ValueError("No sample records provided.")

  attributes_dict = dict()
  attributes = (
      None if attributes is None else set(attributes)  # pyrefly: ignore[bad-assignment]
  )  # for faster lookup

  for feature in sample_records[0].features.feature:
    if attributes is not None and feature not in attributes:
      continue
    data_type = _get_data_type(sample_records[0].features.feature[feature])
    if data_type == dataset_descriptor.DataType.UNSUPPORTED:
      continue
    attributes_dict[feature] = data_type

  for record in sample_records:
    _validate_record(record, attributes_dict)

  dataset_attributes = [
      dataset_descriptor.AttributeDescriptor(
          name=attribute, data_type=data_type
      )
      for attribute, data_type in attributes_dict.items()
  ]
  return dataset_descriptor.DatasetDescriptor(
      attributes=dataset_attributes,
      data_record_converter=TFRecordConverter(attributes_dict),
  )


def _validate_record(
    record: tf.train.Example,
    attributes: dict[str, dataset_descriptor.DataType],
) -> None:
  """Validates a TFRecord against the dataset descriptor.

    We validate that the record has all the attributes in the dataset descriptor
    and that records contain single values of the expected type.

  Args:
    record: A TFRecord record.
    attributes: A dictionary of attribute names and their corresponding data
      types for the dataset.

  Raises:
    ValueError: If the record is not valid.
  """
  num_supported_attributes = 0
  for feature in record.features.feature:
    data_type = _get_data_type(record.features.feature[feature])
    if data_type == dataset_descriptor.DataType.UNSUPPORTED:
      continue

    if feature not in attributes:
      continue

    if data_type != attributes[feature]:
      raise ValueError(
          f"Record has feature {feature} with type {data_type}, but the dataset"
          f" descriptor expects type {attributes[feature]}. A feature should"
          " have the same data type across all records."
      )

    num_values = len(_get_values(feature, record, attributes))
    if num_values > 1:
      raise ValueError(
          f"Record has feature {feature} with {num_values} values."
      )
    num_supported_attributes += 1

  if num_supported_attributes != len(attributes):
    raise ValueError(
        "Record has different attributes from the dataset descriptor."
    )


def _get_data_type(feature: tf.train.Feature) -> dataset_descriptor.DataType:
  """Returns the data type of the feature."""
  if feature.HasField("int64_list"):
    return dataset_descriptor.DataType.INT
  if feature.HasField("bytes_list"):
    return dataset_descriptor.DataType.STR
  if feature.HasField("float_list"):
    return dataset_descriptor.DataType.FLOAT
  return dataset_descriptor.DataType.UNSUPPORTED


def _get_values(
    attribute: str,
    record: tf.train.Example,
    attributes: dict[str, dataset_descriptor.DataType],
) -> list[int | bytes | float]:
  """Returns the values of the attribute in the record."""
  if attributes[attribute] == dataset_descriptor.DataType.INT:
    return record.features.feature[attribute].int64_list.value
  if attributes[attribute] == dataset_descriptor.DataType.STR:
    return record.features.feature[attribute].bytes_list.value
  if attributes[attribute] == dataset_descriptor.DataType.FLOAT:
    return record.features.feature[attribute].float_list.value
  return []

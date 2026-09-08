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

"""Tests for protobuf adapter."""

from __future__ import annotations

import importlib

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth import domain
from dpsynth.adapters import protobuf
from google.protobuf import descriptor_pool
from google.protobuf import message_factory

_desc_module_name = "google.protobuf.descriptor" + "_pb2"
_descriptor_pb2 = importlib.import_module(_desc_module_name)


def _build_test_messages():
  """Constructs test protobuf message classes dynamically."""
  fdp = _descriptor_pb2.FieldDescriptorProto
  file_proto = _descriptor_pb2.FileDescriptorProto(
      name="test_sample.proto", package="test_package"
  )

  enum_proto = file_proto.enum_type.add(name="Status")
  for name, num in [
      ("UNKNOWN", 0),
      ("PENDING", 1),
      ("ACTIVE", 2),
      ("COMPLETED", 3),
  ]:
    enum_proto.value.add(name=name, number=num)

  msg_proto = file_proto.message_type.add(name="FlatUser")
  msg_proto.field.add(name="user_id", number=1, type=fdp.TYPE_INT64)
  msg_proto.field.add(name="username", number=2, type=fdp.TYPE_STRING)
  msg_proto.field.add(
      name="status",
      number=3,
      type=fdp.TYPE_ENUM,
      type_name=".test_package.Status",
  )
  msg_proto.field.add(name="score", number=4, type=fdp.TYPE_FLOAT)
  msg_proto.field.add(name="is_verified", number=5, type=fdp.TYPE_BOOL)

  nested_proto = file_proto.message_type.add(name="NestedMessage")
  nested_proto.field.add(
      name="user",
      number=1,
      type=fdp.TYPE_MESSAGE,
      type_name=".test_package.FlatUser",
  )

  rep_proto = file_proto.message_type.add(name="RepeatedMessage")
  rep_proto.field.add(
      name="values", number=1, type=fdp.TYPE_INT32, label=fdp.LABEL_REPEATED
  )

  unsupp_proto = file_proto.message_type.add(name="UnsupportedFieldMessage")
  unsupp_proto.field.add(name="raw_data", number=1, type=fdp.TYPE_BYTES)

  non_num_proto = file_proto.message_type.add(name="NonNumericalMessage")
  non_num_proto.field.add(name="username", number=1, type=fdp.TYPE_STRING)

  pool = descriptor_pool.DescriptorPool()
  file_desc = pool.Add(file_proto)
  return tuple(
      message_factory.GetMessageClass(file_desc.message_types_by_name[name])
      for name in [
          "FlatUser",
          "NestedMessage",
          "RepeatedMessage",
          "UnsupportedFieldMessage",
          "NonNumericalMessage",
      ]
  )


_DEFAULT_BOUNDS = {"user_id": (0.0, 100.0), "score": (0.0, 100.0)}


class ProtobufAdapterTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    (
        cls.flat_user_cls,
        cls.nested_msg_cls,
        cls.repeated_msg_cls,
        cls.unsupported_msg_cls,
        cls.non_numerical_msg_cls,
    ) = _build_test_messages()

  def test_infer_domain_flat_user(self):
    attributes = protobuf.infer_domain_from_proto(
        self.flat_user_cls, numerical_bounds=_DEFAULT_BOUNDS
    )

    self.assertEqual(
        attributes["user_id"],
        domain.NumericalAttribute(min_value=0.0, max_value=100.0, dtype="int"),
    )
    self.assertEqual(
        attributes["username"], domain.OpenSetCategoricalAttribute()
    )
    self.assertEqual(
        attributes["status"],
        domain.CategoricalAttribute(
            possible_values=["UNKNOWN", "PENDING", "ACTIVE", "COMPLETED"],
            out_of_domain_index=0,
        ),
    )
    self.assertEqual(
        attributes["score"],
        domain.NumericalAttribute(
            min_value=0.0, max_value=100.0, dtype="float"
        ),
    )
    self.assertEqual(
        attributes["is_verified"],
        domain.CategoricalAttribute(
            possible_values=[False, True], out_of_domain_index=0
        ),
    )

  def test_infer_schema(self):
    schema = protobuf.infer_schema(
        self.flat_user_cls, numerical_bounds=_DEFAULT_BOUNDS
    )
    self.assertIsInstance(schema, domain.Schema)
    self.assertLen(schema, 5)
    self.assertEqual(schema["username"], domain.OpenSetCategoricalAttribute())

  def test_input_type_flexibility(self):
    for proto_input in [
        self.flat_user_cls,
        self.flat_user_cls(),
        self.flat_user_cls.DESCRIPTOR,
    ]:
      schema = protobuf.infer_schema(
          proto_input, numerical_bounds=_DEFAULT_BOUNDS
      )
      self.assertIsInstance(schema, domain.Schema)
      self.assertLen(schema, 5)

  def test_invalid_input_type_raises(self):
    with self.assertRaisesRegex(TypeError, "Expected a Protobuf Message class"):
      protobuf.infer_schema("not a proto")  # pyrefly: ignore[bad-argument-type]

  def test_enum_format_number(self):
    attributes = protobuf.infer_domain(
        self.flat_user_cls,
        numerical_bounds=_DEFAULT_BOUNDS,
        enum_format="number",
    )
    self.assertEqual(
        attributes["status"],
        domain.CategoricalAttribute(
            possible_values=[0, 1, 2, 3], out_of_domain_index=0
        ),
    )

  def test_invalid_enum_format_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown enum_format"):
      protobuf.infer_domain(
          self.flat_user_cls,
          numerical_bounds=_DEFAULT_BOUNDS,
          enum_format="other",  # pyrefly: ignore[bad-argument-type]
      )

  def test_numerical_bounds(self):
    bounds = {"user_id": (1.0, 1000.0), "score": (-5.0, 5.0)}
    attributes = protobuf.infer_domain(
        self.flat_user_cls,
        numerical_bounds=bounds,
    )
    self.assertEqual(
        attributes["user_id"],
        domain.NumericalAttribute(min_value=1.0, max_value=1000.0, dtype="int"),
    )
    self.assertEqual(
        attributes["score"],
        domain.NumericalAttribute(min_value=-5.0, max_value=5.0, dtype="float"),
    )

  def test_missing_bounds_raises(self):
    with self.assertRaisesRegex(
        ValueError, "Numerical bounds must be specified for field 'user_id'."
    ):
      protobuf.infer_domain(self.flat_user_cls)

    with self.assertRaisesRegex(
        ValueError, "Numerical bounds must be specified for field 'score'."
    ):
      protobuf.infer_domain(
          self.flat_user_cls,
          numerical_bounds={"user_id": (0.0, 100.0)},
      )

  def test_non_numerical_proto_without_bounds(self):
    attributes = protobuf.infer_domain(self.non_numerical_msg_cls)
    self.assertIn("username", attributes)
    self.assertIsInstance(
        attributes["username"], domain.OpenSetCategoricalAttribute
    )

  def test_nested_message_rejected(self):
    with self.assertRaisesRegex(ValueError, "Nested message field 'user'"):
      protobuf.infer_domain(self.nested_msg_cls)

    attrs = protobuf.infer_domain(
        self.nested_msg_cls, ignore_unsupported_fields=True
    )
    self.assertEmpty(attrs)

  def test_repeated_field_rejected(self):
    with self.assertRaisesRegex(ValueError, "Repeated field 'values'"):
      protobuf.infer_domain(self.repeated_msg_cls)

    attrs = protobuf.infer_domain(
        self.repeated_msg_cls, ignore_unsupported_fields=True
    )
    self.assertEmpty(attrs)

  def test_unsupported_field_rejected(self):
    with self.assertRaisesRegex(ValueError, "Field 'raw_data' of type"):
      protobuf.infer_domain(self.unsupported_msg_cls)

    attrs = protobuf.infer_domain(
        self.unsupported_msg_cls,
        ignore_unsupported_fields=True,
    )
    self.assertEmpty(attrs)

  def test_to_tuple(self):
    user = self.flat_user_cls(
        user_id=42,
        username="alice",
        status=2,  # ACTIVE
        score=95.5,
        is_verified=True,
    )
    t = protobuf.to_tuple(user)
    self.assertEqual(t, (42, "alice", "ACTIVE", 95.5, True))

  def test_to_tuple_number_enum(self):
    user = self.flat_user_cls(
        user_id=42,
        username="alice",
        status=2,
        score=95.5,
        is_verified=True,
    )
    t = protobuf.to_tuple(user, enum_format="number")
    self.assertEqual(t, (42, "alice", 2, 95.5, True))

  def test_from_tuple(self):
    t = (42, "alice", "ACTIVE", 95.5, True)
    user = protobuf.from_tuple(t, self.flat_user_cls)
    self.assertIsInstance(user, self.flat_user_cls)
    self.assertEqual(user.user_id, 42)
    self.assertEqual(user.username, "alice")
    self.assertEqual(user.status, 2)
    self.assertEqual(user.score, 95.5)
    self.assertTrue(user.is_verified)

  def test_from_tuple_from_instance_and_desc(self):
    t = (42, "alice", 2, 95.5, True)
    user1 = protobuf.from_tuple(t, self.flat_user_cls())
    user2 = protobuf.from_tuple(t, self.flat_user_cls.DESCRIPTOR)
    self.assertEqual(user1.user_id, 42)
    self.assertEqual(user2.user_id, 42)

  def test_tuple_proto_roundtrip(self):
    original_tuple = (100, "bob", "PENDING", 12.5, False)
    user = protobuf.from_tuple(original_tuple, self.flat_user_cls)
    roundtripped = protobuf.to_tuple(user)
    self.assertEqual(roundtripped, original_tuple)


if __name__ == "__main__":
  absltest.main()

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

import enum
import math
from typing import Literal

from absl.testing import absltest
from dpsynth import domain
from dpsynth import pydantic_api
import pydantic


class Color(enum.Enum):
  RED = "red"
  GREEN = "green"
  BLUE = "blue"


class ModelForNumericalDefaults(pydantic.BaseModel):
  value_ge_le: int = pydantic.Field(ge=10, le=20)
  value_gt_lt: int = pydantic.Field(gt=10, lt=20)
  value_float_ge_le: float = pydantic.Field(ge=10.0, le=20.0)
  value_float_gt_lt: float = pydantic.Field(gt=10.0, lt=20.0)
  optional_value: int | None = pydantic.Field(ge=10, le=20)


class ModelMissingBounds(pydantic.BaseModel):
  no_bounds_int: int
  partial_bounds_int: int = pydantic.Field(ge=0)


class ModelForCategorical(pydantic.BaseModel):
  is_active: bool
  user_type: Color
  status_code: Literal["A", "B", "C"]
  optional_flag: bool | None
  optional_user_type: Color | None
  optional_status_code: Literal["X", "Y"] | None


class ModelWithUnsupportedType(pydantic.BaseModel):
  complex_field: complex


class SupportedModel(pydantic.BaseModel):
  age: int = pydantic.Field(ge=0, le=100)
  height: float = pydantic.Field(gt=0.0, lt=3.0)
  is_member: bool
  favorite_color: Color
  optional_code: int | None = pydantic.Field(ge=100, le=200)
  literal_status: Literal["active", "inactive"]
  optional_bool: bool | None
  optional_enum: Color | None
  optional_literal: Literal["a", "b"] | None


class PydanticTest(absltest.TestCase):

  def test_get_base_type(self):
    self.assertEqual(pydantic_api._get_base_type(int), (False, int))
    self.assertEqual(pydantic_api._get_base_type(float), (False, float))
    self.assertEqual(pydantic_api._get_base_type(str), (False, str))
    self.assertEqual(pydantic_api._get_base_type(bool), (False, bool))
    self.assertEqual(pydantic_api._get_base_type(Color), (False, Color))
    self.assertEqual(
        pydantic_api._get_base_type(Literal["a", "b"]),
        (False, Literal["a", "b"]),
    )

  def test_get_base_type_optional(self):
    self.assertEqual(pydantic_api._get_base_type(int | None), (True, int))
    self.assertEqual(
        pydantic_api._get_base_type(Literal["x"] | None),
        (True, Literal["x"]),
    )
    self.assertEqual(pydantic_api._get_base_type(Color | None), (True, Color))

  def test_get_base_type_invalid(self):
    with self.assertRaisesRegex(ValueError, "Unexpected None type annotation"):
      pydantic_api._get_base_type(type(None))
    with self.assertRaisesRegex(
        ValueError, "Unexpected type annotation: <class 'list'>"
    ):
      pydantic_api._get_base_type(list)
    with self.assertRaisesRegex(
        AssertionError, "Union must have exactly one non-None type."
    ):
      pydantic_api._get_base_type(int | str | None)

  def test_numerical_attribute_from_field_info(self):
    field_info_int = ModelForNumericalDefaults.model_fields["value_ge_le"]
    attr_int = pydantic_api._numerical_attribute_from_field_info(field_info_int)
    self.assertEqual(
        attr_int,
        domain.NumericalAttribute(
            min_value=10, max_value=20, clip_to_range=True, dtype="int"
        ),
    )

    field_info_float_gt_lt = ModelForNumericalDefaults.model_fields[
        "value_float_gt_lt"
    ]
    attr_float_gt_lt = pydantic_api._numerical_attribute_from_field_info(
        field_info_float_gt_lt
    )
    self.assertEqual(
        attr_float_gt_lt,
        domain.NumericalAttribute(
            min_value=math.nextafter(10.0, math.inf),
            max_value=math.nextafter(20.0, -math.inf),
            clip_to_range=True,
            dtype="float",
        ),
    )

  def test_numerical_attribute_missing_bounds(self):
    field_info_no_bounds = ModelMissingBounds.model_fields["no_bounds_int"]
    with self.assertRaisesRegex(
        ValueError, "Must specify lower and upper bounds for numeric fields."
    ):
      pydantic_api._numerical_attribute_from_field_info(field_info_no_bounds)

  def test_categorical_attribute_from_field_info(self):
    field_info_bool = ModelForCategorical.model_fields["is_active"]
    attr_bool = pydantic_api._categorical_attribute_from_field_info(
        field_info_bool
    )
    self.assertEqual(
        attr_bool,
        domain.CategoricalAttribute(
            possible_values=[False, True], out_of_domain_index=0
        ),
    )

    field_info_enum_opt = ModelForCategorical.model_fields["optional_user_type"]
    attr_enum_opt = pydantic_api._categorical_attribute_from_field_info(
        field_info_enum_opt
    )
    self.assertEqual(
        attr_enum_opt,
        domain.CategoricalAttribute(
            possible_values=[None, Color.RED, Color.GREEN, Color.BLUE],
            out_of_domain_index=0,
        ),
    )

    field_info_literal = ModelForCategorical.model_fields["status_code"]
    attr_literal = pydantic_api._categorical_attribute_from_field_info(
        field_info_literal
    )
    self.assertEqual(
        attr_literal,
        domain.CategoricalAttribute(
            possible_values=["A", "B", "C"], out_of_domain_index=0
        ),
    )

  def test_infer_domain_from_model(self):
    domain_spec = pydantic_api.infer_domain_from_model(SupportedModel)
    expected_domain_spec = {
        "age": domain.NumericalAttribute(
            min_value=0, max_value=100, clip_to_range=True, dtype="int"
        ),
        "height": domain.NumericalAttribute(
            min_value=math.nextafter(0.0, math.inf),
            max_value=math.nextafter(3.0, -math.inf),
            clip_to_range=True,
            dtype="float",
        ),
        "is_member": domain.CategoricalAttribute(
            possible_values=[False, True], out_of_domain_index=0
        ),
        "favorite_color": domain.CategoricalAttribute(
            possible_values=[Color.RED, Color.GREEN, Color.BLUE],
            out_of_domain_index=0,
        ),
        "optional_code": domain.NumericalAttribute(
            min_value=100, max_value=200, clip_to_range=False, dtype="int"
        ),
        "literal_status": domain.CategoricalAttribute(
            possible_values=["active", "inactive"], out_of_domain_index=0
        ),
        "optional_bool": domain.CategoricalAttribute(
            possible_values=[None, False, True], out_of_domain_index=0
        ),
        "optional_enum": domain.CategoricalAttribute(
            possible_values=[None, Color.RED, Color.GREEN, Color.BLUE],
            out_of_domain_index=0,
        ),
        "optional_literal": domain.CategoricalAttribute(
            possible_values=[None, "a", "b"], out_of_domain_index=0
        ),
    }
    self.assertEqual(domain_spec, expected_domain_spec)

  def test_infer_domain_from_model_unsupported_type(self):
    class ModelWithStr(pydantic.BaseModel):
      name: str

    with self.assertRaisesRegex(
        ValueError, "Unexpected type annotation: <class 'str'>"
    ):
      pydantic_api.infer_domain_from_model(ModelWithStr)

    with self.assertRaisesRegex(
        ValueError, "Unexpected type annotation: <class 'complex'>"
    ):
      pydantic_api.infer_domain_from_model(ModelWithUnsupportedType)

  def test_dp_synthetic_data_generation_with_supported_model(self):
    num_records = 1000
    epsilon = 1.0
    delta = 1e-7

    base_real_data = [
        SupportedModel(
            age=30,
            height=1.75,
            is_member=True,
            favorite_color=Color.BLUE,
            optional_code=150,
            literal_status="active",
            optional_bool=True,
            optional_enum=Color.GREEN,
            optional_literal="a",
        ),
        SupportedModel(
            age=22,
            height=1.60,
            is_member=False,
            favorite_color=Color.RED,
            optional_code=None,
            literal_status="inactive",
            optional_bool=None,
            optional_enum=None,
            optional_literal=None,
        ),
    ]
    real_data = (base_real_data * (num_records // len(base_real_data) + 1))[
        :num_records
    ]

    synthetic_records = pydantic_api.dp_synthetic_data_generation(
        data=real_data,
        epsilon=epsilon,
        delta=delta,
    )

    self.assertIsInstance(synthetic_records, list)
    self.assertIsInstance(synthetic_records[0], SupportedModel)

  def test_dp_synthetic_data_generation_with_numerical_model(self):
    num_records = 1000
    epsilon = 1.0
    delta = 1e-7

    base_real_data = [
        ModelForNumericalDefaults(
            value_ge_le=15,
            value_gt_lt=15,
            value_float_ge_le=15.0,
            value_float_gt_lt=15.0,
            optional_value=12,
        ),
        ModelForNumericalDefaults(
            value_ge_le=20,
            value_gt_lt=19,
            value_float_ge_le=10.0,
            value_float_gt_lt=10.1,
            optional_value=None,
        ),
    ]
    real_data = (base_real_data * (num_records // len(base_real_data) + 1))[
        :num_records
    ]

    synthetic_records = pydantic_api.dp_synthetic_data_generation(
        data=real_data,
        epsilon=epsilon,
        delta=delta,
    )

    self.assertIsInstance(synthetic_records, list)
    self.assertIsInstance(synthetic_records[0], ModelForNumericalDefaults)

  def test_dp_synthetic_data_generation_with_categorical_model(self):
    num_records = 1000
    epsilon = 1.0
    delta = 1e-7

    base_real_data = [
        ModelForCategorical(
            is_active=True,
            user_type=Color.GREEN,
            status_code="A",
            optional_flag=False,
            optional_user_type=Color.RED,
            optional_status_code="X",
        ),
        ModelForCategorical(
            is_active=False,
            user_type=Color.BLUE,
            status_code="C",
            optional_flag=None,
            optional_user_type=None,
            optional_status_code=None,
        ),
    ]
    real_data = (base_real_data * (num_records // len(base_real_data) + 1))[
        :num_records
    ]

    synthetic_records = pydantic_api.dp_synthetic_data_generation(
        data=real_data,
        epsilon=epsilon,
        delta=delta,
    )

    self.assertIsInstance(synthetic_records, list)
    self.assertIsInstance(synthetic_records[0], ModelForCategorical)


if __name__ == "__main__":
  absltest.main()

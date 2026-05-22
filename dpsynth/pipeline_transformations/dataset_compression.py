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

"""Transformations to compress each column of the dataset."""

import copy

from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import marginals_computations
from dpsynth.pipeline_transformations import types
import pipeline_dp


def compress_dataset(
    data: types.Collection[tuple[int, ...]],
    backend: pipeline_dp.PipelineBackend,
    dp_engine: pipeline_dp.DPEngine,
    descriptor: types.Collection[dataset_descriptor.DatasetDescriptor],
    num_attributes: int,
) -> tuple[
    types.Collection[tuple[int, ...]],
    types.Collection[dataset_descriptor.DatasetDescriptor],
]:
  """Compresses each column of the dataset.

  Compression is merging rare values into a single ("Other") value.

  Args:
    data: The dataset to compress.
    backend: The pipeline backend to use.
    dp_engine: The DP engine to compute DP marginals.
    descriptor: The dataset descriptor.
    num_attributes: The number of attributes in the dataset.

  Returns:
    A tuple of the compressed dataset and the updated descriptor. The output
    descriptor contains the one way DP marginals by each column which is
    necessary and sufficient for compression/uncompression.
  """

  domain = backend.map(descriptor, lambda x: x.encoded_domain, 'Encoded domain')
  dp_marginals = marginals_computations.compute_one_way_dp_marginals(
      backend,
      dp_engine,
      data,
      domain,
      num_attributes,
  )  # singleton collection of (mbi.LinearMeasurement,)

  def add_linear_measurements_to_descriptor_fn(linear_measurements, descriptor):
    # In Beam we should not modify the input.
    assert len(descriptor.attributes) == len(linear_measurements)
    descriptor_copy = copy.deepcopy(descriptor)
    del descriptor
    for measurement in linear_measurements:
      attribute = descriptor_copy.attributes[measurement.clique[0]]
      attribute.measurement = measurement

    return descriptor_copy

  updated_descriptor = backend.map_with_side_inputs(
      dp_marginals,
      add_linear_measurements_to_descriptor_fn,
      [descriptor],
      'Add measurements to descriptors',
  )  # singleton of (dataset_descriptor.DatasetDescriptor,)

  compressed_data = backend.map_with_side_inputs(
      data,
      lambda row, descriptor: descriptor.compress(row),
      [updated_descriptor],
      'Compress data',
  )  # (tuple[int, ...]])

  return compressed_data, updated_descriptor


def uncompress_dataset(
    data: types.Collection[tuple[int, ...]],
    backend: pipeline_dp.PipelineBackend,
    descriptor: types.Collection[dataset_descriptor.DatasetDescriptor],
) -> types.Collection[tuple[int, ...]]:
  """Uncompresses each column of the dataset."""
  return backend.map_with_side_inputs(
      data,
      lambda row, descriptor: descriptor.uncompress(row),
      [descriptor],
      'Uncompress',
  )

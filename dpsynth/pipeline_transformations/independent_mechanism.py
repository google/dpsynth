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

"""Implementation of the independent mechanism.

In the independent mechanism, we assume that the attributes are independent.
"""

from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.pipeline_transformations import types
import mbi
import pipeline_dp


def fit_model(
    backend: pipeline_dp.PipelineBackend,
    descriptor: types.Collection[dataset_descriptor.DatasetDescriptor],
) -> types.Collection[mbi.MarkovRandomField]:
  """Fits the model."""

  def create_model(
      descriptor: dataset_descriptor.DatasetDescriptor,
  ):
    domain = descriptor.compressed_domain
    marginals = list(descriptor.compressed_measurements())
    # This is not always correct for independent mechanism.
    return mbi.estimation.mirror_descent(domain, marginals, iters=100)

  return backend.map(
      descriptor,
      create_model,
      'Create model for independent mechanism',
  )

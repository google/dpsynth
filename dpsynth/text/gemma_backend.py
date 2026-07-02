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

"""Local Gemma sampling backend for text generation.

``GemmaSamplerBackend`` implements the ``TextGenerationBackend`` protocol using
a local Gemma model via ``gm.text.ChatSampler``.  It supports both pretrained
and DP fine-tuned (LoRA) models.

Example usage::

  from dpsynth.text import gemma_backend, model

  variant = model.GemmaModel.default('gemma3_270m_it')
  backend = gemma_backend.GemmaSamplerBackend.from_lora(
      variant, lora_params=training_state.params,
  )
  results = backend.generate(['Write a haiku about privacy.'])
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

from absl import logging
from dpsynth.text import model
from gemma import gm
from gemma import peft
import pandas as pd
import pydantic


@dataclasses.dataclass(frozen=True)
class GemmaSamplerBackend:
  """``TextGenerationBackend`` using a local Gemma model with ChatSampler.

  Attributes:
    model_variant: Which Gemma model to use.
    lora_config: LoRA configuration (rank, dtype).
    lora_params: Trained LoRA params to merge with pretrained weights.
    max_out_length: Maximum tokens to generate per prompt.
    cache_length: KV-cache length for the sampler.
  """

  model_variant: model.GemmaModel
  lora_config: model.LoraConfig = dataclasses.field(
      default_factory=model.LoraConfig
  )
  lora_params: model.Params | None = None
  max_out_length: int = 512
  cache_length: int = 2048

  @classmethod
  def from_pretrained(cls, variant: model.GemmaModel, **kwargs):
    """Creates a backend from a pretrained model (no LoRA)."""
    return cls(model_variant=variant, **kwargs)

  @classmethod
  def from_lora(
      cls,
      variant: model.GemmaModel,
      lora_params: model.Params,
      lora_config: model.LoraConfig | None = None,
      **kwargs,
  ):
    """Creates a backend from a pretrained model + trained LoRA params."""
    return cls(
        model_variant=variant,
        lora_params=lora_params,
        lora_config=lora_config or model.LoraConfig(),
        **kwargs,
    )

  def _build_sampler(self):
    """Loads the model and returns a ChatSampler."""
    module, frozen_params, init_lora = model.load_gemma(
        self.model_variant,
        self.lora_config,
    )
    # Default to randomly-initialized LoRA; override with trained params.
    lora = init_lora
    if self.lora_params is not None:
      lora = self.lora_params
    params = peft.merge_params(frozen_params, lora)
    logging.info('Built ChatSampler for %s.', self.model_variant.model_class)
    return gm.text.ChatSampler(
        model=module,
        params=params,
        max_out_length=self.max_out_length,
        cache_length=self.cache_length,
    )

  def generate(self, prompts: Sequence[str]) -> list[str]:
    """Generate free-form text from prompts.

    Args:
      prompts: Fully constructed prompts.

    Returns:
      List of exactly ``len(prompts)`` strings. Empty string on failure.
    """
    sampler = self._build_sampler()
    results: list[str] = []
    for i, prompt in enumerate(prompts):
      try:
        results.append(sampler.chat(prompt))
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            'Generation failed for prompt %d: %s', i, e, exc_info=True
        )
        results.append('')
    logging.info('Generated %d/%d prompts.', len(results), len(prompts))
    return results

  def annotate(
      self,
      texts: Sequence[str],
      schema: type[pydantic.BaseModel],
      system_prompt: str,
  ) -> pd.DataFrame:
    """Not supported for local Gemma models."""
    raise NotImplementedError(
        'GemmaSamplerBackend does not support structured annotation. '
        'Use GenAIBackend for constrained decoding.'
    )

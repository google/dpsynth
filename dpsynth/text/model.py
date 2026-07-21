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

"""Gemma model loading and SFT loss function."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, Literal

from absl import logging
from gemma import gm
from gemma import peft
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Type alias for Gemma model params (nested dict of jax.Array).
Params = Any

# Supported model names for GemmaModel.default().
ModelName = Literal[
    'gemma3_270m_it',
    'gemma3_1b_it',
    'gemma3_4b_it',
    'gemma4_e2b_it',
    'gemma4_e4b_it',
]

_DEFAULTS: dict[str, tuple[Callable[..., Any], str, Callable[..., Any]]] = {
    'gemma3_270m_it': (
        gm.nn.Gemma3_270M,
        gm.ckpts.CheckpointPath.GEMMA3_270M_IT,
        gm.text.Gemma3Tokenizer,
    ),
    'gemma3_1b_it': (
        gm.nn.Gemma3_1B,
        gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
        gm.text.Gemma3Tokenizer,
    ),
    'gemma3_4b_it': (
        gm.nn.Gemma3_4B,
        gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
        gm.text.Gemma3Tokenizer,
    ),
    'gemma4_e2b_it': (
        gm.nn.Gemma4_E2B,
        gm.ckpts.CheckpointPath.GEMMA4_E2B_IT,
        gm.text.Gemma4Tokenizer,
    ),
    'gemma4_e4b_it': (
        gm.nn.Gemma4_E4B,
        gm.ckpts.CheckpointPath.GEMMA4_E4B_IT,
        gm.text.Gemma4Tokenizer,
    ),
}


@dataclasses.dataclass(frozen=True)
class GemmaModel:
  """Specification for a Gemma model variant."""

  model_class: Callable[..., Any]
  checkpoint_path: str
  tokenizer_class: Callable[..., Any]

  @classmethod
  def default(cls, name: ModelName) -> GemmaModel:
    """Constructs a GemmaModel from a preset name."""
    if name not in _DEFAULTS:
      raise ValueError(f'Unknown model {name!r}. Options: {list(_DEFAULTS)}')
    model_class, checkpoint_path, tokenizer_class = _DEFAULTS[name]
    return cls(model_class, checkpoint_path, tokenizer_class)


@dataclasses.dataclass(frozen=True)
class LoraConfig:
  """Configuration for LoRA adaptation."""

  rank: int = 16
  dtype: Any = jnp.bfloat16


def load_gemma(
    model_variant: GemmaModel,
    lora_config: LoraConfig,
    *,
    seq_length: int = 64,
) -> tuple[Any, Params, Params]:
  """Loads a pretrained Gemma model with LoRA adapters.

  Args:
    model_variant: Which Gemma variant to load.
    lora_config: LoRA adapter configuration.
    seq_length: Sequence length for model initialization.

  Returns:
    ``(module, frozen_params, trainable_params)`` tuple.
  """
  base_model = model_variant.model_class()
  model = gm.nn.LoRA(
      rank=lora_config.rank,
      model=base_model,
      dtype=lora_config.dtype,
  )

  dummy_tokens = jnp.ones((1, seq_length), dtype=jnp.int32)
  variables = model.init(jax.random.key(0), tokens=dummy_tokens)

  params, lora_params = peft.split_params(variables['params'])
  pt_params = gm.ckpts.load_params(model_variant.checkpoint_path, params=params)

  num_trainable = optax.tree.size(lora_params)
  num_frozen = optax.tree.size(pt_params)
  logging.info(
      'Loaded Gemma model w/ LoRA (rank=%d): %d trainable (%.4f%%), %d frozen',
      lora_config.rank,
      num_trainable,
      100.0 * num_trainable / (num_trainable + num_frozen),
      num_frozen,
  )

  return model, pt_params, lora_params


def sft_loss_fn(
    module: Any,
    full_params: Params,
    data: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Cross-entropy next-token-prediction loss for supervised fine-tuning.

  Args:
    module: LoRA-wrapped Gemma model.
    full_params: Full parameter dict (frozen + trainable, merged).
    data: Dict with ``'input_tokens'`` and ``'loss_mask'`` (int32 ``[B, L]``).

  Returns:
    ``(loss, aux)`` where ``aux`` contains ``'loss'``.
  """
  input_tokens = data['input_tokens']
  loss_mask = data['loss_mask']

  out = module.apply({'params': full_params}, tokens=input_tokens)
  logits = out.logits[:, :-1, :]
  targets = input_tokens[:, 1:]
  mask = loss_mask[:, 1:]

  pt_losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
  loss = jnp.sum(pt_losses * mask) / jnp.maximum(jnp.sum(mask), 1.0)
  return loss, {'loss': loss}


def tokenize_texts(
    examples: Sequence[tuple[str, str]],
    model_variant: GemmaModel,
    max_seq_length: int,
) -> dict[str, np.ndarray]:
  """Tokenizes (prompt, response) pairs for supervised fine-tuning.

  Prompt tokens are masked out (``loss_mask=0``) so only the response
  contributes to the training loss. Turn formatting follows the Gemma
  dialog template (forked from ``gemma/gm/data/_tasks.py``).

  Args:
    examples: Sequence of ``(prompt, response)`` string pairs.
    model_variant: Determines which tokenizer and turn format to use.
    max_seq_length: Maximum sequence length (including special tokens).

  Returns:
    Dict with ``'input_tokens'`` and ``'loss_mask'`` (int32 ``[N, L]``).
  """
  tokenizer = model_variant.tokenizer_class()
  sp = tokenizer.special_tokens
  sot = tokenizer.tokens[sp.START_OF_TURN]
  eot = tokenizer.tokens[sp.END_OF_TURN]

  tokens = np.zeros((len(examples), max_seq_length), dtype=np.int32)
  mask = np.zeros((len(examples), max_seq_length), dtype=np.int32)

  for i, (prompt, response) in enumerate(examples):
    # Embed turn tags as strings so SentencePiece handles tokenization
    # boundaries correctly (encoding pieces separately can shift BPE merges).
    prompt_str = f'{sot}user\n{prompt}{eot}\n{sot}model\n'
    response_str = f'{response}{eot}'
    prompt_ids = tokenizer.encode(prompt_str, add_bos=True)
    response_ids = tokenizer.encode(response_str, add_eos=True)

    ids = prompt_ids + response_ids
    length = min(len(ids), max_seq_length)
    tokens[i, :length] = ids[:length]
    # Mask: 0 for prompt, 1 for response.
    resp_start = min(len(prompt_ids), length)
    mask[i, resp_start:length] = 1

  logging.info(
      'Tokenized %d examples (max_seq_length=%d)',
      len(examples),
      max_seq_length,
  )

  return {'input_tokens': tokens, 'loss_mask': mask}

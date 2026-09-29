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
import itertools
import time
from typing import Any, Literal

from absl import logging
from etils import epath
from gemma import gm
from gemma import peft
import jax
import jax.numpy as jnp
from kauldron import kd
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
    checkpoint_path: epath.PathLike | None = None,
    sharding: Any = None,
) -> tuple[gm.nn.LoRA, Params, Params]:
  """Loads a Gemma model with LoRA adapters.

  Args:
    model_variant: Which Gemma variant to load.
    lora_config: LoRA adapter configuration.
    checkpoint_path: Checkpoint to restore from. If None, loads pretrained base
      weights from ``model_variant.checkpoint_path`` and initializes fresh LoRA
      adapters. If a checkpoint is provided, restores base and LoRA params.
    sharding: Optional sharding tree to constrain parameters across devices.

  Returns:
    ``(module, base_params, lora_params)`` tuple.
  """
  base_model = model_variant.model_class()
  model = gm.nn.LoRA(
      rank=lora_config.rank,
      model=base_model,
      dtype=lora_config.dtype,
  )
  if checkpoint_path is not None:
    all_params = gm.ckpts.load_params(checkpoint_path, sharding=sharding)
    # LoRA params are present when loading from a lora checkpoint
    base_params, lora_params = peft.split_params(all_params)  # pyrefly: ignore[bad-argument-type]
  else:
    # When starting from scratch, initialize LoRA params
    init_params = model.init(
        jax.random.key(0),
        tokens=jnp.ones((1, 64), dtype=jnp.int32),
    )['params']
    _, lora_params = peft.split_params(init_params)
    lora_params = kd.sharding.with_sharding_constraint(lora_params, sharding)
    base_params = gm.ckpts.load_params(
        model_variant.checkpoint_path, sharding=sharding
    )

  num_lora = optax.tree.size(lora_params)
  num_base = optax.tree.size(base_params)
  logging.info(
      'Loaded Gemma model w/ LoRA (rank=%d): %d lora (%.4f%%), %d base',
      lora_config.rank,
      num_lora,
      100.0 * num_lora / (num_lora + num_base),
      num_base,
  )
  return model, base_params, lora_params


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


def format_prompt(prompt: str, tokenizer: Any) -> str:
  """Formats a prompt string with the Gemma user and model turn tags."""
  sp = tokenizer.special_tokens
  sot, eot = (
      tokenizer.tokens[sp.START_OF_TURN],
      tokenizer.tokens[sp.END_OF_TURN],
  )
  # Universal chat prompt format expected by Gemma instruction-tuned models.
  return f'{sot}user\n{prompt}{eot}\n{sot}model\n'


def format_response(response: str, tokenizer: Any) -> str:
  """Formats a model response string with the end-of-turn tag."""
  eot = tokenizer.tokens[tokenizer.special_tokens.END_OF_TURN]
  return f'{response}{eot}'


def tokenize_example(
    example: tuple[str, str],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, np.ndarray]:
  """Tokenizes a single (prompt, response) pair for supervised fine-tuning.

  Prompt tokens are masked out (``loss_mask=0``) so only the response
  contributes to the training loss.

  Args:
    example: ``(prompt, response)`` string pair.
    tokenizer: Gemma tokenizer instance.
    max_seq_length: Maximum sequence length (including special tokens).

  Returns:
    Dict with ``'input_tokens'`` and ``'loss_mask'`` (int32 ``[L]``).
  """
  prompt, response = example
  prompt_str = format_prompt(prompt, tokenizer)
  response_str = format_response(response, tokenizer)
  prompt_ids = tokenizer.encode(prompt_str, add_bos=True)
  response_ids = tokenizer.encode(response_str, add_eos=True)

  ids = prompt_ids + response_ids
  length = min(len(ids), max_seq_length)
  tokens = np.zeros(max_seq_length, dtype=np.int32)
  mask = np.zeros(max_seq_length, dtype=np.int32)
  tokens[:length] = ids[:length]
  mask[min(len(prompt_ids), length) : length] = 1

  return {'input_tokens': tokens, 'loss_mask': mask}


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

  tokens = np.zeros((len(examples), max_seq_length), dtype=np.int32)
  mask = np.zeros((len(examples), max_seq_length), dtype=np.int32)

  for i, example in enumerate(examples):
    tokenized = tokenize_example(example, tokenizer, max_seq_length)
    tokens[i] = tokenized['input_tokens']
    mask[i] = tokenized['loss_mask']

  logging.info(
      'Tokenized %d examples (max_seq_length=%d)',
      len(examples),
      max_seq_length,
  )

  return {'input_tokens': tokens, 'loss_mask': mask}


class GemmaSampler:
  """Batched inference sampler for generating synthetic text."""

  def __init__(
      self,
      *,
      model: Any,
      params: Params,
      max_seq_length: int,
      temperature: float,
  ):
    sampling_method = (
        gm.text.RandomSampling(temperature=temperature)
        if temperature > 0
        else gm.text.Greedy()
    )
    self._sampler = gm.text.Sampler(
        model=model,
        params=params,
        cache_length=max_seq_length,
        max_out_length=max_seq_length,
        sampling=sampling_method,
    )

  def __call__(
      self,
      prompts: Sequence[str],
      *,
      rng: int = 0,
      batch_size: int = 32,
  ) -> list[str]:
    """Formats prompts, batches them across devices, and samples responses.

    Args:
      prompts: Sequence of prompt instruction strings.
      rng: Base random seed for sampling (default 0). The random seed is set per
        batch (``rng + batch_idx``), so fixing the seed and changing the
        ``batch_size`` will change the sampled outputs.
      batch_size: Inference batch size (default 32).

    Returns:
      List of generated response strings corresponding to each prompt.
    """
    formatted = [format_prompt(p, self._sampler.tokenizer) for p in prompts]

    results: list[str] = []
    for i, batch_items in enumerate(itertools.batched(formatted, batch_size)):
      cur_size = len(batch_items)
      batch = list(batch_items) + [batch_items[-1]] * (batch_size - cur_size)
      t0 = time.perf_counter()
      responses = self._sampler.sample(
          batch, sharding=kd.sharding.FIRST_DIM, rng=rng + i
      )
      elapsed = time.perf_counter() - t0
      logging.info(
          'Batch %d: %d samples in %.2fs (%.2f samples/s)',
          i + 1,
          cur_size,
          elapsed,
          cur_size / elapsed if elapsed > 0 else 0.0,
      )
      results.extend([str(r) for r in responses[:cur_size]])
    return results

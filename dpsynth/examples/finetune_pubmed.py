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

r"""Differentially private fine-tuning of Gemma on PubMed abstracts.

Fine-tunes a small Gemma model on real biomedical abstracts with DP-BandMF, then
samples *synthetic* abstracts from it. Every training example uses the same
fixed instruction prompt, so the model learns the unconditional distribution of
abstracts; we generate by feeding that prompt back with a non-zero temperature.

Training is the expensive, privacy-consuming step; sampling is free
post-processing. The --train / --sample flags let you train once and then draw
more synthetic data later without retraining. This example is for demonstration
purposes only: it's not super highly tuned, and is missing some of the bells
and whistles of a production job.
"""

import json
import typing

from absl import app
from absl import flags
from absl import logging
import datasets
from dpsynth.text import dp_sft
from dpsynth.text import model
from etils import epath
from gemma import gm
from gemma import peft
from jax_privacy import execution_plan
import optax

_MODEL = flags.DEFINE_enum(
    'model',
    'gemma3_270m_it',
    [
        'gemma3_270m_it',
        'gemma3_1b_it',
        'gemma3_4b_it',
        'gemma4_e2b_it',
        'gemma4_e4b_it',
    ],
    'Gemma model variant to fine-tune.',
)
_WORKDIR = flags.DEFINE_string(
    'workdir', '', 'Directory to save the fine-tuned model to and load it from.'
)
_TRAIN = flags.DEFINE_bool('train', True, 'Whether to DP-fine-tune the model.')
_SAMPLE = flags.DEFINE_bool(
    'sample', True, 'Whether to generate synthetic abstracts.'
)
_ITERATIONS = flags.DEFINE_integer('iterations', 200, 'DP-SGD iterations.')

_MICROBATCH_SIZE = flags.DEFINE_integer(
    'microbatch_size', 8, 'Microbatch size for DP-SGD.'
)
_EPSILON = flags.DEFINE_float('epsilon', 8.0, 'Target DP epsilon.')
_DELTA = flags.DEFINE_float('delta', 1e-5, 'Target DP delta.')
_LORA_RANK = flags.DEFINE_integer('lora_rank', 16, 'LoRA rank.')
_MAX_SEQ_LENGTH = flags.DEFINE_integer('max_seq_length', 512, 'Max tokens.')
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 1e-4, 'AdamW learning rate.'
)
_NUM_SAMPLES = flags.DEFINE_integer('num_samples', 8, 'Abstracts to generate.')
_MAX_OUT_LENGTH = flags.DEFINE_integer(
    'max_out_length', 512, 'Max output tokens.'
)
_TEMPERATURE = flags.DEFINE_float('temperature', 1.0, 'Sampling temperature.')
_SEED = flags.DEFINE_integer(
    'seed', 0, 'RNG seed for sampling; change it to draw a different batch.'
)
_DATA_PATH = flags.DEFINE_string(
    'data_path',
    '',
    'Optional path to a JSON list of abstract strings. When empty, the '
    'abstracts are downloaded from the HuggingFace Hub instead.',
)

_INSTRUCTION = 'Write the abstract of a research paper.'
_DATASET = 'ccdv/pubmed-summarization'


def _model_variant() -> model.GemmaModel:
  # ``_MODEL`` is a DEFINE_enum whose choices are exactly ``model.ModelName``.
  return model.GemmaModel.default(typing.cast(model.ModelName, _MODEL.value))


def _load_abstracts() -> list[str]:
  """Loads abstracts from --data_path (JSON) or the HF Hub."""
  if _DATA_PATH.value:
    with epath.Path(_DATA_PATH.value).open() as f:
      raw = json.load(f)
      if raw and isinstance(raw[0], dict):
        abstracts = [d['abstract'] for d in raw if 'abstract' in d]
      else:
        abstracts = raw
  else:
    data = datasets.load_dataset(
        _DATASET, split='train', trust_remote_code=True
    )
    abstracts = data['abstract']
  return [t.strip() for t in abstracts if t and t.strip()]


def train() -> dp_sft.FineTuneResult:
  """DP-fine-tunes Gemma on PubMed abstracts."""
  abstracts = _load_abstracts()
  # Frame each abstract as a (fixed instruction prompt -> abstract) SFT pair.
  train_data = [(_INSTRUCTION, abstract) for abstract in abstracts]

  fine_tuner = dp_sft.DPFineTuner(
      model_variant=_model_variant(),
      mechanism_config=execution_plan.BandMFConfig.default(
          num_bands=8,
          iterations=_ITERATIONS.value,
          expected_participations=8.0,
      ),
      lora_rank=_LORA_RANK.value,
      max_seq_length=_MAX_SEQ_LENGTH.value,
      optimizer=optax.adamw(_LEARNING_RATE.value),
      performance_flags=execution_plan.PerformanceFlags(
          microbatch_size=_MICROBATCH_SIZE.value
      ),
  ).calibrate(epsilon=_EPSILON.value, delta=_DELTA.value)

  logging.info('DP fine-tuning on %d abstracts...', len(train_data))
  return fine_tuner(rng=0, data=train_data)


def load_finetuned() -> dp_sft.FineTuneResult:
  """Rebuilds the model and loads previously saved fine-tuned params.

  Uses the same --model / --lora_rank / --max_seq_length as training, so the
  reconstructed architecture matches the saved parameters.

  Returns:
    A FineTuneResult holding the reconstructed model and loaded params.
  """
  module, frozen, trainable = model.load_gemma(
      _model_variant(),
      model.LoraConfig(rank=_LORA_RANK.value),
      seq_length=_MAX_SEQ_LENGTH.value,
  )
  template = peft.merge_params(frozen, trainable)
  params = gm.ckpts.load_params(_ckpt_dir(), params=template)
  return dp_sft.FineTuneResult(model=module, params=params)


def sample(result: dp_sft.FineTuneResult) -> None:
  """Generates synthetic abstracts and writes them to <workdir> as JSONL."""
  sampler = gm.text.ChatSampler(
      model=result.model,
      params=typing.cast(typing.Mapping[str, typing.Any], result.params),
      max_out_length=_MAX_OUT_LENGTH.value,
      sampling=gm.text.RandomSampling(temperature=_TEMPERATURE.value),
  )
  out_path = epath.Path(_WORKDIR.value) / 'synthetic_abstracts.jsonl'
  with out_path.open('w') as f:
    # One abstract per call is simple but slow. For higher throughput, pass a
    # *list* to the batched sampler (`sampler.sampler.sample` for Gemma 3,
    # `sampler.gemma4_sampler.sample` for Gemma 4); chunk it to fit HBM.
    for i in range(_NUM_SAMPLES.value):
      abstract = sampler.chat(_INSTRUCTION, rng=_SEED.value + i)
      logging.info('Synthetic abstract %d:\n%s', i + 1, abstract)
      f.write(json.dumps({'abstract': abstract}) + '\n')
  logging.info('Wrote %d abstracts to %s.', _NUM_SAMPLES.value, out_path)


def _ckpt_dir() -> epath.Path:
  """Path to store the final fine-tuned params."""
  return epath.Path(_WORKDIR.value) / 'params'


def main(_) -> None:
  if not _WORKDIR.value:
    raise app.UsageError('--workdir is required.')
  if not _TRAIN.value and not _SAMPLE.value:
    raise app.UsageError('Set at least one of --train / --sample.')

  if _TRAIN.value:
    result = train()
    ckpt_path = _ckpt_dir()
    if ckpt_path.exists():
      logging.info(
          'Checkpoint directory %s already exists; removing before save.',
          ckpt_path,
      )
      ckpt_path.rmtree()
    gm.ckpts.save_params(
        typing.cast(typing.Mapping[str, typing.Any], result.params),
        ckpt_path,
        wait_until_finished=True,
    )
    logging.info('Saved fine-tuned model to %s.', ckpt_path)
  else:
    result = load_finetuned()

  if _SAMPLE.value:
    sample(result)


if __name__ == '__main__':
  app.run(main)

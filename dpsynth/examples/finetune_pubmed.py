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

This example demonstrates the end-to-end ``DPFineTuner`` workflow for generating
differentially private synthetic text:

  1. Load real biomedical abstracts from the PubMed summarization benchmark.
  2. Frame each abstract as a supervised fine-tuning example: a fixed
     instruction prompt -> the abstract text.
  3. Differentially privately fine-tune a small Gemma model with DP-SGD.
  4. Generate *synthetic* abstracts by sampling from the fine-tuned model.

The end result is a generator of biomedical-paper abstracts that reflects the
aggregate style and content of the training corpus while satisfying a formal
(epsilon, delta)-DP guarantee with respect to the individual papers.

Because every training example shares the same fixed instruction prompt, the
model learns the *unconditional* distribution of abstracts: at generation time
we feed that same prompt and sample multiple times (with a non-zero
temperature) to draw diverse synthetic abstracts.

This is a long-running job (model loading + DP-SGD + sampling on an
accelerator), and hence is provided as a binary rather than a colab notebook.

Example (single GPU/TPU host):

  python -m dpsynth.examples.finetune_pubmed \
      --iterations=200 \
      --epsilon=8.0
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
import datasets
from dpsynth.text import dp_sft
from dpsynth.text import model
from gemma import gm
import jax
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
_ITERATIONS = flags.DEFINE_integer(
    'iterations', 200, 'Number of DP-SGD training iterations.'
)
_EPSILON = flags.DEFINE_float(
    'epsilon', 8.0, 'Target (epsilon, delta)-DP epsilon.'
)
_DELTA = flags.DEFINE_float('delta', 1e-5, 'Target (epsilon, delta)-DP delta.')
_LORA_RANK = flags.DEFINE_integer('lora_rank', 16, 'LoRA rank.')
_MAX_SEQ_LENGTH = flags.DEFINE_integer(
    'max_seq_length', 512, 'Max token sequence length per example.'
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 1e-4, 'AdamW learning rate.'
)
_NUM_SAMPLES = flags.DEFINE_integer(
    'num_samples', 8, 'Number of synthetic abstracts to generate at the end.'
)
_MAX_OUT_LENGTH = flags.DEFINE_integer(
    'max_out_length', 512, 'Max tokens to generate per synthetic abstract.'
)
_TEMPERATURE = flags.DEFINE_float(
    'temperature',
    1.0,
    'Sampling temperature for synthetic abstract generation.',
)

# A fixed instruction prompt prepended to every training example. Because it is
# identical for all examples, the model learns to produce an abstract whenever
# it sees this prompt -- i.e. an unconditional abstract generator.
_INSTRUCTION = 'Write the abstract of a biomedical research paper.'

# Standard scientific-abstract benchmark. DP-SGD needs a large corpus for good
# utility, so we train on the full (~119k-abstract) PubMed train split.
_DATASET = 'ccdv/pubmed-summarization'
_SPLIT = 'train'


def load_abstracts() -> list[str]:
  """Loads all abstracts from the PubMed summarization benchmark train split.

  Returns:
    A list of every non-empty abstract string in the split.
  """
  logging.info('Loading %s [%s] via Hugging Face datasets...', _DATASET, _SPLIT)
  # ``trust_remote_code`` is required because the dataset ships a loading
  # script.
  dataset = datasets.load_dataset(
      _DATASET, split=_SPLIT, trust_remote_code=True
  )
  abstracts = [
      text.strip() for text in dataset['abstract'] if text and text.strip()
  ]
  logging.info('Loaded %d abstracts.', len(abstracts))
  return abstracts


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Devices (%d): %s', len(jax.devices()), jax.devices())

  # 1. Load real biomedical abstracts from the PubMed benchmark.
  abstracts = load_abstracts()
  if not abstracts:
    raise app.UsageError('No abstracts were loaded from the dataset.')

  # 2. Frame each abstract as a (fixed instruction prompt -> abstract) SFT pair.
  train_data = [(_INSTRUCTION, abstract) for abstract in abstracts]

  # 3. Differentially privately fine-tune Gemma with DP-SGD.
  model_name: model.ModelName = _MODEL.value  # type: ignore[assignment]
  model_variant = model.GemmaModel.default(model_name)
  config = execution_plan.BandMFConfig.default(
      num_bands=1,
      iterations=_ITERATIONS.value,
      expected_participations=1.0,
  )
  fine_tuner = dp_sft.DPFineTuner(
      model_variant=model_variant,
      mechanism_config=config,
      lora_rank=_LORA_RANK.value,
      max_seq_length=_MAX_SEQ_LENGTH.value,
      optimizer=optax.adamw(_LEARNING_RATE.value),
      performance_flags=execution_plan.PerformanceFlags(microbatch_size=1),
  ).calibrate(epsilon=_EPSILON.value, delta=_DELTA.value)

  logging.info(
      'DP fine-tuning on %d abstracts for %d iterations at (eps=%.1f, '
      'delta=%.0e)...',
      len(train_data),
      _ITERATIONS.value,
      _EPSILON.value,
      _DELTA.value,
  )
  result = fine_tuner(rng=0, data=train_data)
  logging.info('Fine-tuning complete.')

  # 4. Generate synthetic abstracts by sampling the fine-tuned model. We use the
  # same instruction prompt with a non-zero temperature and a different rng per
  # draw to obtain diverse outputs.
  sampler = gm.text.ChatSampler(
      model=result.model,
      params=result.params,
      max_out_length=_MAX_OUT_LENGTH.value,
      sampling=gm.text.RandomSampling(temperature=_TEMPERATURE.value),
  )

  logging.info('Generating %d synthetic abstracts...', _NUM_SAMPLES.value)
  for i in range(_NUM_SAMPLES.value):
    synthetic_abstract = sampler.chat(_INSTRUCTION, rng=i)
    logging.info('=' * 70)
    logging.info('SYNTHETIC ABSTRACT %d:\n%s', i + 1, synthetic_abstract)
  logging.info('=' * 70)
  logging.info('Done. Generated %d synthetic abstracts.', _NUM_SAMPLES.value)


if __name__ == '__main__':
  app.run(main)

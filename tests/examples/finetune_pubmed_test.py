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

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from dpsynth.examples import finetune_pubmed
from dpsynth.text import dp_sft
from etils import epath
from gemma import gm
from kauldron import kd


class MockBatchedSampler:

  def __init__(self):
    self.calls = []

  def sample(self, prompt_batch, *, sharding, rng):
    self.calls.append((list(prompt_batch), sharding, rng))
    return [f'Abstract {len(self.calls)}_{i}' for i in range(len(prompt_batch))]


class MockChatSamplerGemma3:

  def __init__(self, *unused_args, **unused_kwargs):
    self._sampler = MockBatchedSampler()

  @property
  def sampler(self):
    return self._sampler

  @property
  def gemma4_sampler(self):
    raise AttributeError('gemma4_sampler not available')


class MockChatSamplerGemma4:

  def __init__(self, *unused_args, **unused_kwargs):
    self._gemma4_sampler = MockBatchedSampler()

  @property
  def sampler(self):
    raise AttributeError('sampler not available')

  @property
  def gemma4_sampler(self):
    return self._gemma4_sampler


class FinetunePubmedTest(absltest.TestCase):

  def test_sample_batches_and_padding_gemma3(self):
    mock_chat = MockChatSamplerGemma3()
    workdir = self.create_tempdir()

    with flagsaver.flagsaver(
        workdir=workdir.full_path,
        batch_size=4,
        num_samples=10,
        seed=123,
    ):
      with mock.patch.object(gm.text, 'ChatSampler', return_value=mock_chat):
        dummy_result = dp_sft.FineTuneResult(
            model=mock.MagicMock(), params=mock.MagicMock()
        )
        finetune_pubmed.sample(dummy_result)

    calls = mock_chat.sampler.calls
    self.assertLen(calls, 3)

    # Check batch lengths (undersized final batch padded to batch_size 4).
    self.assertLen(calls[0][0], 4)
    self.assertLen(calls[1][0], 4)
    self.assertLen(calls[2][0], 4)

    # Check sharding parameter.
    self.assertEqual(calls[0][1], kd.sharding.FIRST_DIM)
    self.assertEqual(calls[1][1], kd.sharding.FIRST_DIM)
    self.assertEqual(calls[2][1], kd.sharding.FIRST_DIM)

    # Check output JSONL file.
    out_path = epath.Path(workdir.full_path) / 'synthetic_abstracts.jsonl'
    self.assertTrue(out_path.exists())
    lines = out_path.read_text().strip().splitlines()
    self.assertLen(lines, 10)
    for line in lines:
      data = json.loads(line)
      self.assertIn('abstract', data)

  def test_sample_batches_and_padding_gemma4(self):
    mock_chat = MockChatSamplerGemma4()
    workdir = self.create_tempdir()

    with flagsaver.flagsaver(
        workdir=workdir.full_path,
        batch_size=4,
        num_samples=6,
        seed=42,
    ):
      with mock.patch.object(gm.text, 'ChatSampler', return_value=mock_chat):
        dummy_result = dp_sft.FineTuneResult(
            model=mock.MagicMock(), params=mock.MagicMock()
        )
        finetune_pubmed.sample(dummy_result)

    calls = mock_chat.gemma4_sampler.calls
    self.assertLen(calls, 2)
    self.assertLen(calls[0][0], 4)
    self.assertLen(calls[1][0], 4)

    out_path = epath.Path(workdir.full_path) / 'synthetic_abstracts.jsonl'
    lines = out_path.read_text().strip().splitlines()
    self.assertLen(lines, 6)


if __name__ == '__main__':
  absltest.main()

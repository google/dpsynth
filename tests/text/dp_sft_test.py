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

"""Tests for dpsynth.text.model and dpsynth.text.dp_sft.

These tests verify the loss function, configuration, and calibration logic
without loading any real model checkpoints.
"""

import dataclasses
import math
from unittest import mock

from absl.testing import absltest
from dpsynth.text import dp_sft
from dpsynth.text import dp_trainer
from dpsynth.text import model
from gemma import peft
import grain.python as pygrain
import jax
import jax.numpy as jnp
from jax_privacy import execution_plan
import numpy as np


def _default_config():
  return execution_plan.BandMFConfig.default(
      num_bands=1,
      iterations=100,
      expected_participations=1.0,
  )


def _make_tiny_model_and_params(vocab_size=8, embed_dim=4, seq_len=4):
  """Creates a minimal Flax Linen model and random params for testing."""
  import flax.linen as nn  # pylint: disable=g-import-not-at-top

  @dataclasses.dataclass
  class _Output:
    logits: jnp.ndarray

  class TinyModel(nn.Module):
    vocab_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, tokens, **kwargs):
      x = nn.Embed(self.vocab_size, self.embed_dim)(tokens)
      logits = nn.Dense(self.vocab_size)(x)
      return _Output(logits=logits)

  module = TinyModel(vocab_size=vocab_size, embed_dim=embed_dim)
  dummy = jnp.ones((1, seq_len), dtype=jnp.int32)
  params = module.init(jax.random.key(0), dummy)['params']
  return module, params


class SftLossFnTest(absltest.TestCase):

  def test_loss_is_finite_and_positive(self):
    module, params = _make_tiny_model_and_params()
    data = {
        'input_tokens': jnp.array([[1, 2, 3, 4, 0]], dtype=jnp.int32),
        'loss_mask': jnp.array([[1, 1, 1, 1, 0]], dtype=jnp.int32),
    }
    loss, aux = model.sft_loss_fn(module, params, data)
    self.assertTrue(jnp.isfinite(loss))
    self.assertGreater(float(loss), 0.0)
    self.assertIn('loss', aux)

  def test_loss_decreases_with_gradient_descent(self):
    """Gradient descent on a tiny model should reduce loss."""
    module, params = _make_tiny_model_and_params()
    tokens = jnp.array([[1, 2, 3, 1]], dtype=jnp.int32)
    mask = jnp.ones_like(tokens)
    data = {'input_tokens': tokens, 'loss_mask': mask}

    loss_before = float(model.sft_loss_fn(module, params, data)[0])

    # Overfit on this single example.
    for _ in range(200):

      def step_fn(p):
        return model.sft_loss_fn(module, p, data)

      (_, _), grads = jax.value_and_grad(step_fn, has_aux=True)(params)
      params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)

    loss_after = float(model.sft_loss_fn(module, params, data)[0])
    self.assertLess(loss_after, loss_before)

  def test_masked_tokens_do_not_contribute(self):
    module, params = _make_tiny_model_and_params()
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    mask_short = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)
    mask_full = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)

    loss_short = float(
        model.sft_loss_fn(
            module, params, {'input_tokens': tokens, 'loss_mask': mask_short}
        )[0]
    )
    loss_full = float(
        model.sft_loss_fn(
            module, params, {'input_tokens': tokens, 'loss_mask': mask_full}
        )[0]
    )
    # Different masks should give different losses.
    self.assertNotAlmostEqual(loss_short, loss_full, places=3)


class GemmaModelTest(absltest.TestCase):

  def test_default_preset(self):
    m = model.GemmaModel.default('gemma3_270m_it')
    self.assertIsNotNone(m.checkpoint_path)
    self.assertTrue(callable(m.model_class))
    self.assertTrue(callable(m.tokenizer_class))

  def test_custom_checkpoint_path(self):
    m = dataclasses.replace(
        model.GemmaModel.default('gemma3_270m_it'),
        checkpoint_path='/custom/path',
    )
    self.assertEqual(m.checkpoint_path, '/custom/path')

  def test_unknown_name_raises(self):
    with self.assertRaises(ValueError):
      model.GemmaModel.default('nonexistent')


class LoraConfigTest(absltest.TestCase):

  def test_defaults(self):
    cfg = model.LoraConfig()
    self.assertEqual(cfg.rank, 16)
    self.assertEqual(cfg.dtype, jnp.bfloat16)

  def test_custom_values(self):
    cfg = model.LoraConfig(rank=8, dtype=jnp.float32)
    self.assertEqual(cfg.rank, 8)
    self.assertEqual(cfg.dtype, jnp.float32)


class DPFineTunerTest(absltest.TestCase):

  def test_creates_valid_mechanism_config(self):
    mechanism = dp_sft.DPFineTuner(
        model_variant=model.GemmaModel.default('gemma3_270m_it'),
        mechanism_config=_default_config(),
    )
    self.assertEqual(mechanism.mechanism_config.iterations, 100)
    self.assertIsNone(mechanism.mechanism_config.noise_multiplier)

  def test_calibrate_sets_noise_multiplier(self):
    mechanism = dp_sft.DPFineTuner(
        model_variant=model.GemmaModel.default('gemma3_270m_it'),
        mechanism_config=_default_config(),
    ).configure(zcdp_rho=0.5)
    self.assertIsNotNone(mechanism.mechanism_config.noise_multiplier)
    self.assertGreater(mechanism.mechanism_config.noise_multiplier, 0.0)

  def test_calibrate_noise_formula(self):
    mechanism = dp_sft.DPFineTuner(
        model_variant=model.GemmaModel.default('gemma3_270m_it'),
        mechanism_config=_default_config(),
    ).configure(zcdp_rho=0.5)
    # Single band: rounds = iterations, sigma = sqrt(T / (2*rho)).
    expected = math.sqrt(100 / (2.0 * 0.5))
    self.assertAlmostEqual(
        mechanism.mechanism_config.noise_multiplier, expected
    )

  def test_dp_event_before_calibration_raises(self):
    mechanism = dp_sft.DPFineTuner(
        model_variant=model.GemmaModel.default('gemma3_270m_it'),
        mechanism_config=_default_config(),
    )
    with self.assertRaises(ValueError):
      _ = mechanism.dp_event

  def test_dp_event_after_calibration(self):
    mechanism = dp_sft.DPFineTuner(
        model_variant=model.GemmaModel.default('gemma3_270m_it'),
        mechanism_config=_default_config(),
    ).configure(zcdp_rho=0.5)
    event = mechanism.dp_event
    self.assertIsNotNone(event)

  @mock.patch.object(peft, 'merge_params', autospec=True)
  @mock.patch.object(dp_trainer, 'DPTrainer', autospec=True)
  @mock.patch.object(model, 'load_gemma', autospec=True)
  def test_call_with_map_dataset_materializes_to_numpy(
      self, mock_load, mock_trainer_cls, mock_merge
  ):
    mock_load.return_value = (mock.MagicMock(), {}, {})
    mock_trainer = mock_trainer_cls.return_value
    mock_trainer.return_value = mock.MagicMock(params={})
    mock_merge.return_value = {}

    variant = model.GemmaModel(
        model_class=mock.MagicMock(),
        checkpoint_path='/mock/path',
        tokenizer_class=_MockTokenizer,
    )
    pairs = [('prompt1', 'resp1'), ('prompt2', 'resp2')]
    ds = pygrain.MapDataset.source(pairs)

    fine_tuner = dp_sft.DPFineTuner(
        model_variant=variant,
        mechanism_config=_default_config(),
        max_seq_length=16,
    ).configure(zcdp_rho=0.5)

    res = fine_tuner(rng=0, data=ds)
    self.assertIsInstance(res, dp_sft.FineTuneResult)
    mock_trainer.assert_called_once()
    passed_dataset = mock_trainer.call_args.kwargs['data']
    self.assertIsInstance(passed_dataset['input_tokens'], np.ndarray)
    self.assertIsInstance(passed_dataset['loss_mask'], np.ndarray)
    self.assertEqual(passed_dataset['input_tokens'].shape, (2, 16))
    self.assertEqual(passed_dataset['loss_mask'].shape, (2, 16))

    expected_dataset = model.tokenize_texts(pairs, variant, max_seq_length=16)
    np.testing.assert_array_equal(
        passed_dataset['input_tokens'], expected_dataset['input_tokens']
    )
    np.testing.assert_array_equal(
        passed_dataset['loss_mask'], expected_dataset['loss_mask']
    )

  @mock.patch.object(peft, 'merge_params', autospec=True)
  @mock.patch.object(dp_trainer, 'DPTrainer', autospec=True)
  @mock.patch.object(model, 'load_gemma', autospec=True)
  def test_call_with_sequence(self, mock_load, mock_trainer_cls, mock_merge):
    mock_load.return_value = (mock.MagicMock(), {}, {})
    mock_trainer = mock_trainer_cls.return_value
    mock_trainer.return_value = mock.MagicMock(params={})
    mock_merge.return_value = {}

    variant = model.GemmaModel(
        model_class=mock.MagicMock(),
        checkpoint_path='/mock/path',
        tokenizer_class=_MockTokenizer,
    )
    pairs = [('prompt1', 'resp1'), ('prompt2', 'resp2')]
    fine_tuner = dp_sft.DPFineTuner(
        model_variant=variant,
        mechanism_config=_default_config(),
        max_seq_length=16,
    ).configure(zcdp_rho=0.5)

    res = fine_tuner(rng=0, data=pairs)
    self.assertIsInstance(res, dp_sft.FineTuneResult)
    mock_trainer.assert_called_once()
    passed_dataset = mock_trainer.call_args.kwargs['data']
    expected_dataset = model.tokenize_texts(pairs, variant, max_seq_length=16)
    np.testing.assert_array_equal(
        passed_dataset['input_tokens'], expected_dataset['input_tokens']
    )
    np.testing.assert_array_equal(
        passed_dataset['loss_mask'], expected_dataset['loss_mask']
    )


class _MockSpecialTokens:
  START_OF_TURN = 0
  END_OF_TURN = 1


class _MockTokenizer:

  def __init__(self):
    self.special_tokens = _MockSpecialTokens()
    self.tokens = ['<start>', '<end>']

  def encode(self, text, add_bos=False, add_eos=False):
    del add_bos, add_eos
    return [len(text) % 10 + 1, len(text) % 5 + 1]


class TokenizeExampleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tokenizer = _MockTokenizer()

  def test_tokenize_example_shapes_and_types(self):
    example = ('Hello', 'World')
    result = model.tokenize_example(example, self.tokenizer, max_seq_length=16)
    self.assertIn('input_tokens', result)
    self.assertIn('loss_mask', result)
    self.assertEqual(result['input_tokens'].shape, (16,))
    self.assertEqual(result['loss_mask'].shape, (16,))
    self.assertEqual(result['input_tokens'].dtype, np.int32)
    self.assertEqual(result['loss_mask'].dtype, np.int32)

  def test_tokenize_example_masking(self):
    example = ('Hi', 'There')
    result = model.tokenize_example(example, self.tokenizer, max_seq_length=8)
    np.testing.assert_array_equal(result['loss_mask'][:2], [0, 0])
    np.testing.assert_array_equal(result['loss_mask'][2:4], [1, 1])
    np.testing.assert_array_equal(result['loss_mask'][4:], [0, 0, 0, 0])

  def test_tokenize_example_truncation(self):
    example = ('Hi', 'There')
    result = model.tokenize_example(example, self.tokenizer, max_seq_length=3)
    self.assertEqual(result['input_tokens'].shape, (3,))
    self.assertEqual(result['loss_mask'].shape, (3,))
    np.testing.assert_array_equal(result['loss_mask'], [0, 0, 1])


class TokenizeTextsTest(absltest.TestCase):

  def test_parity_with_tokenize_example(self):
    variant = model.GemmaModel(
        model_class=mock.MagicMock(),
        checkpoint_path='/mock/path',
        tokenizer_class=_MockTokenizer,
    )
    tokenizer = variant.tokenizer_class()
    pairs = [('Hello', 'World'), ('How are you?', 'I am fine.')]
    seq_result = model.tokenize_texts(
        pairs, model_variant=variant, max_seq_length=16
    )
    for i, pair in enumerate(pairs):
      ex_result = model.tokenize_example(pair, tokenizer, max_seq_length=16)
      np.testing.assert_array_equal(
          seq_result['input_tokens'][i], ex_result['input_tokens']
      )
      np.testing.assert_array_equal(
          seq_result['loss_mask'][i], ex_result['loss_mask']
      )

  def test_parity_between_sequence_and_numpy_array(self):
    variant = model.GemmaModel(
        model_class=mock.MagicMock(),
        checkpoint_path='/mock/path',
        tokenizer_class=_MockTokenizer,
    )
    pairs = [('Hello', 'World'), ('How are you?', 'I am fine.')]
    seq_result = model.tokenize_texts(
        pairs, model_variant=variant, max_seq_length=16
    )
    np_result = model.tokenize_texts(
        np.asarray(pairs), model_variant=variant, max_seq_length=16
    )

    np.testing.assert_array_equal(
        seq_result['input_tokens'], np_result['input_tokens']
    )
    np.testing.assert_array_equal(
        seq_result['loss_mask'], np_result['loss_mask']
    )


if __name__ == '__main__':
  absltest.main()

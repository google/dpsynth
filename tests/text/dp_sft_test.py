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

import math

from absl.testing import absltest
from dpsynth.text import dp_sft
from dpsynth.text import model
from flax import nnx
import jax
import jax.numpy as jnp


class _TinyModel(nnx.Module):
  """Minimal model that mimics the Gemma forward signature."""

  def __init__(self, vocab_size: int, rngs: nnx.Rngs):
    self.embed = nnx.Linear(vocab_size, vocab_size, rngs=rngs)
    self.vocab_size = vocab_size

  def __call__(self, input_tokens, positions, cache, attention_mask):
    one_hot = jax.nn.one_hot(input_tokens, self.vocab_size)
    logits = self.embed(one_hot)
    return logits, None


class SftLossFnTest(absltest.TestCase):

  def test_loss_is_finite_and_positive(self):
    tiny = _TinyModel(vocab_size=8, rngs=nnx.Rngs(0))
    data = {
        'input_tokens': jnp.array([[1, 2, 3, 4, 0]], dtype=jnp.int32),
        'input_mask': jnp.array([[1, 1, 1, 1, 0]], dtype=jnp.int32),
    }
    loss, aux = model.sft_loss_fn(tiny, data)
    self.assertTrue(jnp.isfinite(loss))
    self.assertGreater(float(loss), 0.0)
    self.assertIn('loss', aux)

  def test_loss_decreases_for_correct_predictions(self):
    """A model that always predicts the right next token should have low loss."""
    vocab_size = 4
    tiny = _TinyModel(vocab_size=vocab_size, rngs=nnx.Rngs(0))
    tokens = jnp.array([[1, 2, 3, 1]], dtype=jnp.int32)
    mask = jnp.ones_like(tokens)
    data = {'input_tokens': tokens, 'input_mask': mask}

    loss_before = float(model.sft_loss_fn(tiny, data)[0])

    # Overfit the tiny model on this single example.
    for _ in range(200):

      def step_fn(m):
        return model.sft_loss_fn(m, data)

      _, grads = nnx.value_and_grad(step_fn, has_aux=True)(tiny)
      state = nnx.state(tiny)
      state = jax.tree.map(lambda p, g: p - 0.1 * g, state, grads)
      nnx.update(tiny, state)

    loss_after = float(model.sft_loss_fn(tiny, data)[0])
    self.assertLess(loss_after, loss_before)

  def test_masked_tokens_do_not_contribute(self):
    tiny = _TinyModel(vocab_size=8, rngs=nnx.Rngs(0))
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    # Only the first two tokens are unmasked.
    mask_short = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)
    mask_full = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)

    loss_short = float(
        model.sft_loss_fn(
            tiny,
            {
                'input_tokens': tokens,
                'input_mask': mask_short,
            },
        )[0]
    )
    loss_full = float(
        model.sft_loss_fn(
            tiny,
            {
                'input_tokens': tokens,
                'input_mask': mask_full,
            },
        )[0]
    )
    # Different masks should give different losses.
    self.assertNotAlmostEqual(loss_short, loss_full, places=3)


class SupportedModelTest(absltest.TestCase):

  def test_gemma3_270m_properties(self):
    m = model.SupportedModel.GEMMA3_270M_PT
    self.assertIsInstance(m.checkpoint_path, str)
    self.assertIn('GEMMA3_270M', m.checkpoint_path)
    self.assertIsNotNone(m.model_config)
    self.assertTrue(callable(m.load_fn))


class LoraConfigTest(absltest.TestCase):

  def test_defaults(self):
    cfg = model.LoraConfig()
    self.assertEqual(cfg.rank, 16)
    self.assertIsNone(cfg.alpha)
    self.assertIsNone(cfg.weight_qtype)

  def test_custom_values(self):
    cfg = model.LoraConfig(rank=8, alpha=32.0, weight_qtype='nf4')
    self.assertEqual(cfg.rank, 8)
    self.assertEqual(cfg.alpha, 32.0)
    self.assertEqual(cfg.weight_qtype, 'nf4')


class DPSftDefaultTest(absltest.TestCase):

  def test_default_creates_valid_config(self):
    mechanism = dp_sft.DPSft.default(
        iterations=100,
        batch_size=8,
        num_examples=1000,
    )
    self.assertEqual(mechanism.config.iterations, 100)
    self.assertIsNone(mechanism.config.noise_multiplier)

  def test_calibrate_sets_noise_multiplier(self):
    mechanism = dp_sft.DPSft.default(
        iterations=100,
        batch_size=8,
        num_examples=1000,
    ).calibrate(zcdp_rho=0.5)
    self.assertIsNotNone(mechanism.config.noise_multiplier)
    self.assertGreater(mechanism.config.noise_multiplier, 0.0)

  def test_calibrate_noise_formula(self):
    mechanism = dp_sft.DPSft.default(
        iterations=100,
        batch_size=8,
        num_examples=1000,
    ).calibrate(zcdp_rho=0.5)
    # Single band: rounds = iterations, sigma = sqrt(T / (2*rho)).
    expected = math.sqrt(100 / (2.0 * 0.5))
    self.assertAlmostEqual(mechanism.config.noise_multiplier, expected)

  def test_dp_event_before_calibration_raises(self):
    mechanism = dp_sft.DPSft.default(
        iterations=100,
        batch_size=8,
        num_examples=1000,
    )
    with self.assertRaises(ValueError):
      _ = mechanism.dp_event

  def test_dp_event_after_calibration(self):
    mechanism = dp_sft.DPSft.default(
        iterations=100,
        batch_size=8,
        num_examples=1000,
    ).calibrate(zcdp_rho=0.5)
    event = mechanism.dp_event
    self.assertIsNotNone(event)


if __name__ == '__main__':
  absltest.main()

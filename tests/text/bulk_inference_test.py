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

from typing import Literal
from unittest import mock

from absl.testing import absltest
from dpsynth.text import bulk_inference
from dpsynth.text import prompts
import pandas as pd
import pydantic

Topic = Literal['Science', 'Technology', 'Other']
Complexity = Literal['Low', 'Medium', 'High']


class SimpleFeatures(pydantic.BaseModel):
  """Test feature schema."""

  topic: Topic = pydantic.Field(description='The main topic.')
  complexity: Complexity = pydantic.Field(description='Complexity level.')


class ModelNameTest(absltest.TestCase):

  def test_gemini_values(self):
    self.assertEqual(
        bulk_inference.ModelName.GEMINI_2_5_FLASH_LITE, 'gemini-2.5-flash-lite'
    )
    self.assertEqual(
        bulk_inference.ModelName.GEMINI_3_5_FLASH, 'gemini-3.5-flash'
    )

  def test_gemma_values(self):
    self.assertEqual(bulk_inference.ModelName.GEMMA_4_27B, 'gemma-4-26b-a4b-it')
    self.assertEqual(bulk_inference.ModelName.GEMMA_4_31B, 'gemma-4-31b-it')


class StripMarkdownFencesTest(absltest.TestCase):

  def test_strips_json_fence(self):
    text = '```json\n{"topic": "Science"}\n```'
    self.assertEqual(
        bulk_inference._strip_markdown_fences(text), '{"topic": "Science"}'
    )

  def test_strips_plain_fence(self):
    text = '```\n{"topic": "Science"}\n```'
    self.assertEqual(
        bulk_inference._strip_markdown_fences(text), '{"topic": "Science"}'
    )

  def test_noop_on_plain_json(self):
    text = '{"topic": "Science"}'
    self.assertEqual(bulk_inference._strip_markdown_fences(text), text)

  def test_strips_whitespace(self):
    text = '  \n{"topic": "Science"}\n  '
    self.assertEqual(
        bulk_inference._strip_markdown_fences(text), '{"topic": "Science"}'
    )


class GenAIBackendAnnotateTest(absltest.TestCase):

  @mock.patch('google.genai.Client')
  def test_annotate_index_aligned_on_success(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_response = mock.MagicMock()
    mock_response.text = '{"topic": "Science", "complexity": "High"}'
    mock_client.models.generate_content.return_value = mock_response

    backend = bulk_inference.GenAIBackend()

    df = backend.annotate(['text1', 'text2'], SimpleFeatures, 'Annotate.')
    self.assertLen(df, 2)
    self.assertListEqual(list(df.columns), ['topic', 'complexity'])

  @mock.patch('google.genai.Client')
  def test_annotate_fills_none_on_failure(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.models.generate_content.side_effect = RuntimeError('API down')

    backend = bulk_inference.GenAIBackend()

    df = backend.annotate(['text1', 'text2', 'text3'], SimpleFeatures, 'Sys.')
    self.assertLen(df, 3)
    self.assertTrue(df.iloc[0].isna().all())
    self.assertTrue(df.iloc[2].isna().all())

  @mock.patch('google.genai.Client')
  def test_annotate_mixed_success_and_failure(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    good = mock.MagicMock()
    good.text = '{"topic": "Technology", "complexity": "Low"}'
    bad = RuntimeError('fail')

    mock_client.models.generate_content.side_effect = [good, bad, good]

    backend = bulk_inference.GenAIBackend()

    df = backend.annotate(['a', 'b', 'c'], SimpleFeatures, 'Sys.')
    self.assertLen(df, 3)
    self.assertEqual(df.iloc[0]['topic'], 'Technology')
    self.assertTrue(pd.isna(df.iloc[1]['topic']))
    self.assertEqual(df.iloc[2]['topic'], 'Technology')

  @mock.patch('google.genai.Client')
  def test_annotate_handles_markdown_fenced_json(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    fenced = mock.MagicMock()
    fenced.text = '```json\n{"topic": "Science", "complexity": "High"}\n```'
    mock_client.models.generate_content.return_value = fenced

    backend = bulk_inference.GenAIBackend()

    df = backend.annotate(['text1'], SimpleFeatures, 'Sys.')
    self.assertLen(df, 1)
    self.assertEqual(df.iloc[0]['topic'], 'Science')
    self.assertEqual(df.iloc[0]['complexity'], 'High')


class GenAIBackendGenerateTest(absltest.TestCase):

  @mock.patch('google.genai.Client')
  def test_generate_index_aligned_on_success(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_response = mock.MagicMock()
    mock_response.text = 'Generated text.'
    mock_client.models.generate_content.return_value = mock_response

    backend = bulk_inference.GenAIBackend()

    results = backend.generate(['prompt1', 'prompt2'])
    self.assertLen(results, 2)
    self.assertEqual(results[0], 'Generated text.')

  @mock.patch('google.genai.Client')
  def test_generate_fills_empty_on_failure(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    good = mock.MagicMock()
    good.text = 'OK'
    mock_client.models.generate_content.side_effect = [
        good,
        RuntimeError('fail'),
        good,
    ]

    backend = bulk_inference.GenAIBackend()

    results = backend.generate(['a', 'b', 'c'])
    self.assertLen(results, 3)
    self.assertEqual(results[0], 'OK')
    self.assertEqual(results[1], '')
    self.assertEqual(results[2], 'OK')


class AnnotateFeaturesPromptTest(absltest.TestCase):

  def test_includes_dataset_description(self):
    prompt = prompts.annotate_features_prompt(
        dataset_description='News articles.',
        dataclass=SimpleFeatures,
        text='Some text.',
    )
    self.assertIn('News articles', prompt)

  def test_includes_feature_names_and_descriptions(self):
    prompt = prompts.annotate_features_prompt(
        dataset_description='Test.',
        dataclass=SimpleFeatures,
        text='Hello.',
    )
    self.assertIn('topic', prompt)
    self.assertIn('complexity', prompt)
    self.assertIn('The main topic', prompt)

  def test_includes_possible_values(self):
    prompt = prompts.annotate_features_prompt(
        dataset_description='Test.',
        dataclass=SimpleFeatures,
        text='Hello.',
    )
    self.assertIn('Science', prompt)
    self.assertIn('Technology', prompt)

  def test_includes_text_to_annotate(self):
    prompt = prompts.annotate_features_prompt(
        dataset_description='Test.',
        dataclass=SimpleFeatures,
        text='Quantum computing advances.',
    )
    self.assertIn('Quantum computing advances', prompt)

  def test_template_placeholder(self):
    prompt = prompts.annotate_features_prompt(
        dataset_description='Test.',
        dataclass=SimpleFeatures,
        text='{{text}}',
    )
    self.assertIn('{{text}}', prompt)


class ConditionalGenerationPromptTest(absltest.TestCase):

  def test_includes_target_features(self):
    prompt = prompts.conditional_generation_prompt(
        dataset_description='Test.',
        target_features='- topic: Science\n- complexity: High',
        formatting_requirements='Be concise.',
    )
    self.assertIn('topic: Science', prompt)
    self.assertIn('complexity: High', prompt)

  def test_includes_formatting_requirements(self):
    prompt = prompts.conditional_generation_prompt(
        dataset_description='Test.',
        target_features='- topic: Tech',
        formatting_requirements='Use bullet points.',
    )
    self.assertIn('Use bullet points', prompt)

  def test_with_exemplar(self):
    prompt = prompts.conditional_generation_prompt(
        dataset_description='Test.',
        target_features='- topic: Science',
        formatting_requirements='Formal.',
        exemplar=({'topic': 'Tech'}, 'Example tech article.'),
    )
    self.assertIn('topic: Tech', prompt)
    self.assertIn('Example tech article', prompt)
    self.assertIn('Completed Example', prompt)

  def test_without_exemplar(self):
    prompt = prompts.conditional_generation_prompt(
        dataset_description='Test.',
        target_features='- topic: Science',
        formatting_requirements='Formal.',
    )
    self.assertNotIn('Completed Example', prompt)


if __name__ == '__main__':
  absltest.main()

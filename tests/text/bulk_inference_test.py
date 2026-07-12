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
from dpsynth import domain
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


# Equivalent dpsynth Domain representation for the same schema.
SIMPLE_DOMAIN: dict[str, domain.AttributeType] = {
    'topic': domain.CategoricalAttribute(
        possible_values=['Science', 'Technology', 'Other'],
        description='The main topic.',
    ),
    'complexity': domain.CategoricalAttribute(
        possible_values=['Low', 'Medium', 'High'],
        description='Complexity level.',
    ),
}

# Domain with mixed attribute types for testing defaults.
MIXED_DOMAIN: dict[str, domain.AttributeType] = {
    'topic': domain.CategoricalAttribute(
        possible_values=['Science', 'Technology', 'Other'],
    ),
    'name': domain.OpenSetCategoricalAttribute(),
    'age': domain.NumericalAttribute(min_value=0, max_value=120),
    'bio': domain.FreeFormTextAttribute(),
}


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


@mock.patch('google.genai.Client', autospec=True)
class GenAIBackendAnnotateTest(absltest.TestCase):
  """Tests for GenAIBackend.annotate."""

  @staticmethod
  def _create_inlined_response(
      text: str | None = None, error: str | None = None
  ) -> mock.MagicMock:
    """Creates a mock inlined batch response."""
    mock_response = mock.MagicMock()
    mock_response.text = text
    return mock.MagicMock(error=error, response=mock_response if text else None)

  @staticmethod
  def _create_mock_job(
      state,
      inlined_responses: list[mock.MagicMock] | None = None,
      done_side_effect: list[bool] | None = None,
      inlined_responses_side_effect: list[list[mock.MagicMock]] | None = None,
      error: str | None = None,
      name: str = 'job',
  ) -> mock.MagicMock:
    """Creates a mock batch job."""
    job = mock.MagicMock()
    job.name = name
    if done_side_effect is not None:
      type(job).done = mock.PropertyMock(side_effect=done_side_effect)
    else:
      type(job).done = mock.PropertyMock(return_value=True)
    job.state = state
    job.error = error

    mock_dest = mock.MagicMock()
    if inlined_responses_side_effect is not None:
      type(mock_dest).inlined_responses = mock.PropertyMock(
          side_effect=inlined_responses_side_effect
      )
    elif inlined_responses is not None:
      mock_dest.inlined_responses = inlined_responses
    job.dest = mock_dest
    return job

  def test_annotate_success(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    inlined_resp = self._create_inlined_response(
        '{"topic": "Science", "complexity": "High"}'
    )
    mock_batch_job = self._create_mock_job(
        state=bulk_inference.types.JobState.JOB_STATE_SUCCEEDED,
        inlined_responses=[inlined_resp, inlined_resp],
    )

    mock_client.batches.create.return_value = mock_batch_job
    mock_client.batches.get.return_value = mock_batch_job

    backend = bulk_inference.GenAIBackend(
        api_key='fake', poll_interval_seconds=0
    )
    df = backend.annotate(['text1', 'text2'], SimpleFeatures, 'Sys.')
    self.assertLen(df, 2)
    self.assertEqual(df.iloc[0]['topic'], 'Science')
    self.assertEqual(df.iloc[1]['topic'], 'Science')

  def test_annotate_raises_on_length_mismatch(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    inlined_resp = self._create_inlined_response(
        '{"topic": "Science", "complexity": "High"}'
    )
    mock_batch_job = self._create_mock_job(
        state=bulk_inference.types.JobState.JOB_STATE_SUCCEEDED,
        inlined_responses=[inlined_resp],
    )

    mock_client.batches.create.return_value = mock_batch_job
    mock_client.batches.get.return_value = mock_batch_job

    backend = bulk_inference.GenAIBackend(
        api_key='fake', poll_interval_seconds=0
    )
    with self.assertRaisesRegex(ValueError, 'got 1 results for 2 inputs'):
      backend.annotate(['text1', 'text2'], SimpleFeatures, 'Sys.')

  def test_annotate_marks_failed_pydantic_rows(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    inlined_error = self._create_inlined_response(error='Failed item')
    inlined_success = self._create_inlined_response(
        '{"topic": "Science", "complexity": "High"}'
    )

    mock_batch_job = self._create_mock_job(
        state=bulk_inference.types.JobState.JOB_STATE_SUCCEEDED,
        inlined_responses=[inlined_error, inlined_success],
    )

    mock_client.batches.create.return_value = mock_batch_job
    mock_client.batches.get.return_value = mock_batch_job

    backend = bulk_inference.GenAIBackend(
        api_key='fake', poll_interval_seconds=0
    )
    df = backend.annotate(['text1', 'text2'], SimpleFeatures, 'Sys.')
    self.assertLen(df, 2)
    self.assertEqual(df.iloc[0]['_fields_decoded'], 0)
    self.assertEqual(df.iloc[1]['_fields_decoded'], 2)
    self.assertTrue(pd.isna(df.iloc[0]['topic']))  # NaN for failed row.
    self.assertEqual(df.iloc[1]['topic'], 'Science')

  def test_annotate_raises_on_failed_job(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_batch_job = self._create_mock_job(
        state=bulk_inference.types.JobState.JOB_STATE_FAILED,
        error='Something went wrong',
    )

    mock_client.batches.create.return_value = mock_batch_job
    mock_client.batches.get.return_value = mock_batch_job

    backend = bulk_inference.GenAIBackend(
        api_key='fake', poll_interval_seconds=0
    )
    with self.assertRaisesRegex(
        RuntimeError,
        'Batch job .* ended with state.* Error: Something went wrong',
    ):
      backend.annotate(['text1'], SimpleFeatures, 'Sys.')

  def test_annotate_respects_class_chunk_size(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    inlined_resp = self._create_inlined_response(
        '{"topic": "Science", "complexity": "High"}'
    )
    mock_batch_job = self._create_mock_job(
        state=bulk_inference.types.JobState.JOB_STATE_SUCCEEDED,
        inlined_responses=[inlined_resp, inlined_resp],
    )

    mock_client.batches.create.return_value = mock_batch_job
    mock_client.batches.get.return_value = mock_batch_job

    backend = bulk_inference.GenAIBackend(
        api_key='fake', poll_interval_seconds=0, chunk_size=2
    )
    # 4 texts, chunk_size=2 -> 2 chunks/jobs
    backend.annotate(['t1', 't2', 't3', 't4'], SimpleFeatures, 'Sys.')
    self.assertEqual(mock_client.batches.create.call_count, 2)

  def test_annotate_respects_max_concurrent_jobs(self, mock_client_cls):
    mock_client = mock_client_cls.return_value
    create_resp = GenAIBackendAnnotateTest._create_inlined_response
    create_job = GenAIBackendAnnotateTest._create_mock_job

    jobs = []

    def create_side_effect(model, src):
      del model  # Unused.
      inlined_resp = create_resp('{"topic": "Science", "complexity": "High"}')
      job = create_job(
          state=bulk_inference.types.JobState.JOB_STATE_SUCCEEDED,
          inlined_responses=[inlined_resp] * len(src),
          done_side_effect=[False, True],
          name=f'job-{len(jobs)}',
      )
      jobs.append(job)
      return job

    mock_client.batches.create.side_effect = create_side_effect

    def get_side_effect(name):
      idx = int(name.split('-')[1])
      return jobs[idx]

    mock_client.batches.get.side_effect = get_side_effect

    backend = bulk_inference.GenAIBackend(
        api_key='fake',
        poll_interval_seconds=0,
        chunk_size=2,
        max_concurrent_jobs=1,
    )
    backend.annotate(['t1', 't2', 't3', 't4', 't5'], SimpleFeatures, 'Sys.')
    self.assertEqual(mock_client.batches.create.call_count, 3)


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


class DomainToJsonSchemaTest(absltest.TestCase):
  """Tests for domain_to_json_schema."""

  def test_categorical_produces_enum(self):
    schema = bulk_inference.domain_to_json_schema(SIMPLE_DOMAIN)
    self.assertEqual(schema['type'], 'object')
    topic_prop = schema['properties']['topic']
    self.assertEqual(topic_prop['type'], 'string')
    self.assertEqual(topic_prop['enum'], ['Science', 'Technology', 'Other'])

  def test_categorical_with_description(self):
    schema = bulk_inference.domain_to_json_schema(SIMPLE_DOMAIN)
    self.assertEqual(
        schema['properties']['topic']['description'], 'The main topic.'
    )

  def test_categorical_rejects_none(self):
    with self.assertRaises(ValueError):
      domain.CategoricalAttribute(possible_values=[None, 'A', 'B'])

  def test_numerical_produces_number(self):
    d = {
        'age': domain.NumericalAttribute(
            min_value=0, max_value=120, description='Person age.'
        ),
    }
    schema = bulk_inference.domain_to_json_schema(d)
    age_prop = schema['properties']['age']
    self.assertEqual(age_prop['type'], 'number')
    self.assertEqual(age_prop['minimum'], 0.0)
    self.assertEqual(age_prop['maximum'], 120.0)
    self.assertEqual(age_prop['description'], 'Person age.')

  def test_open_set_produces_string(self):
    d = {
        'name': domain.OpenSetCategoricalAttribute(description='Person name.'),
    }
    schema = bulk_inference.domain_to_json_schema(d)
    self.assertEqual(schema['properties']['name']['type'], 'string')
    self.assertNotIn('enum', schema['properties']['name'])

  def test_free_form_text_produces_string(self):
    d = {
        'bio': domain.FreeFormTextAttribute(description='A short biography.'),
    }
    schema = bulk_inference.domain_to_json_schema(d)
    self.assertEqual(schema['properties']['bio']['type'], 'string')

  def test_all_fields_required(self):
    schema = bulk_inference.domain_to_json_schema(SIMPLE_DOMAIN)
    self.assertCountEqual(schema['required'], ['topic', 'complexity'])

  def test_integer_categorical_produces_integer_type(self):
    d = {
        'rating': domain.CategoricalAttribute(possible_values=[1, 2, 3, 4, 5]),
    }
    schema = bulk_inference.domain_to_json_schema(d)
    self.assertEqual(schema['properties']['rating']['type'], 'integer')

  def test_boolean_categorical_produces_boolean_type(self):
    d = {
        'active': domain.CategoricalAttribute(possible_values=[True, False]),
    }
    schema = bulk_inference.domain_to_json_schema(d)
    self.assertEqual(schema['properties']['active']['type'], 'boolean')

  def test_float_categorical_produces_number_type(self):
    d = {
        'score': domain.CategoricalAttribute(possible_values=[0.5, 1.0, 1.5]),
    }
    schema = bulk_inference.domain_to_json_schema(d)
    self.assertEqual(schema['properties']['score']['type'], 'number')


class DefaultValueTest(absltest.TestCase):
  """Tests for _default_value."""

  def test_categorical_returns_ood_sentinel(self):
    attr = domain.CategoricalAttribute(
        possible_values=['A', 'B', 'C'], out_of_domain_index=2
    )
    self.assertEqual(bulk_inference._default_value(attr), 'C')

  def test_open_set_returns_default_value(self):
    attr = domain.OpenSetCategoricalAttribute(default_value='<UNK>')
    self.assertEqual(bulk_inference._default_value(attr), '<UNK>')

  def test_numerical_returns_min_value(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    self.assertEqual(bulk_inference._default_value(attr), 0.0)

  def test_free_form_text_returns_empty_string(self):
    attr = domain.FreeFormTextAttribute()
    self.assertEqual(bulk_inference._default_value(attr), '')

  def test_default_row_all_types(self):
    row = bulk_inference._default_row(MIXED_DOMAIN)
    self.assertEqual(row['topic'], 'Science')  # out_of_domain_index=0
    self.assertEqual(row['name'], '<OOD>')
    self.assertEqual(row['age'], 0.0)
    self.assertEqual(row['bio'], '')


@mock.patch('google.genai.Client')
class GenAIBackendDomainAnnotateTest(absltest.TestCase):
  """Tests GenAIBackend.annotate with a dpsynth Domain."""

  def setUp(self):
    super().setUp()
    self.backend = bulk_inference.GenAIBackend(
        api_key='fake', poll_interval_seconds=0
    )

  @staticmethod
  def _inlined(text=None, error=None):
    resp = mock.MagicMock()
    resp.text = text
    return mock.MagicMock(error=error, response=resp if text else None)

  def _mock_job(self, mock_client_cls, inlined_responses):
    job = mock.MagicMock()
    job.name = 'job'
    type(job).done = mock.PropertyMock(return_value=True)
    job.state = bulk_inference.types.JobState.JOB_STATE_SUCCEEDED
    job.error = None
    job.dest = mock.MagicMock(inlined_responses=inlined_responses)
    mock_client_cls.return_value.batches.create.return_value = job
    mock_client_cls.return_value.batches.get.return_value = job
    return job

  def test_success(self, mock_client_cls):
    resp = self._inlined('{"topic": "Science", "complexity": "High"}')
    self._mock_job(mock_client_cls, [resp, resp])
    df = self.backend.annotate(['t1', 't2'], SIMPLE_DOMAIN, 'Sys.')
    self.assertLen(df, 2)
    self.assertListEqual(list(df['_fields_decoded']), [2, 2])
    self.assertEqual(df.iloc[0]['topic'], 'Science')

  def test_error_fills_defaults_and_fields_decoded(self, mock_client_cls):
    bad = self._inlined('not valid json')
    partial = self._inlined('{"topic": "Science"}')
    good = self._inlined('{"topic": "Science", "complexity": "High"}')
    self._mock_job(mock_client_cls, [bad, partial, good])
    df = self.backend.annotate(['t1', 't2', 't3'], SIMPLE_DOMAIN, 'Sys.')
    self.assertLen(df, 3)
    self.assertListEqual(list(df['_fields_decoded']), [0, 1, 2])
    # Malformed → all defaults.
    self.assertEqual(df.iloc[0]['topic'], 'Science')  # OOD sentinel
    self.assertEqual(df.iloc[0]['complexity'], 'Low')  # OOD sentinel
    # Partial → decoded field kept, missing filled.
    self.assertEqual(df.iloc[1]['topic'], 'Science')
    self.assertEqual(df.iloc[1]['complexity'], 'Low')
    # Full success.
    self.assertEqual(df.iloc[2]['complexity'], 'High')

  def test_item_error_fills_defaults(self, mock_client_cls):
    err = self._inlined(error='Failed item')
    good = self._inlined('{"topic": "Science", "complexity": "High"}')
    self._mock_job(mock_client_cls, [err, good])
    df = self.backend.annotate(['t1', 't2'], SIMPLE_DOMAIN, 'Sys.')
    self.assertLen(df, 2)
    self.assertEqual(df.iloc[0]['_fields_decoded'], 0)
    self.assertEqual(df.iloc[1]['_fields_decoded'], 2)

  def test_extra_keys_dropped(self, mock_client_cls):
    resp = self._inlined(
        '{"topic": "Science", "complexity": "High", "extra": "x"}'
    )
    self._mock_job(mock_client_cls, [resp])
    df = self.backend.annotate(['t1'], SIMPLE_DOMAIN, 'Sys.')
    self.assertLen(df, 1)
    self.assertEqual(df.iloc[0]['_fields_decoded'], 2)
    self.assertNotIn('extra', df.columns)

  def test_passes_json_schema_not_pydantic(self, mock_client_cls):
    resp = self._inlined('{"topic": "Science", "complexity": "High"}')
    self._mock_job(mock_client_cls, [resp])
    self.backend.annotate(['t1'], SIMPLE_DOMAIN, 'Sys.')
    create_call = mock_client_cls.return_value.batches.create.call_args
    config = create_call.kwargs['src'][0].config
    self.assertIsInstance(config.response_schema, dict)
    self.assertIn('properties', config.response_schema)


if __name__ == '__main__':
  absltest.main()

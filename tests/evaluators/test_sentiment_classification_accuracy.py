"""
Tests for the sentiment classification accuracy evaluator.
"""

from unittest.mock import Mock, patch

import pytest
from arize.experimental.datasets.experiments.types import EvaluationResult

from arize_experiment.core.task import TaskResult
from arize_experiment.evaluators.sentiment_classification_accuracy import (
    SentimentClassificationAccuracyEvaluator,
)


@pytest.fixture
def mock_openai_response() -> Mock:
    """Create a mock OpenAI API response."""
    mock_message = Mock()
    mock_message.content = "correct This sentiment classification is accurate."

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_completion = Mock()
    mock_completion.choices = [mock_choice]

    return mock_completion


@pytest.fixture
def mock_openai_client(mock_openai_response: Mock) -> Mock:
    """Create a mock OpenAI client."""
    mock_client = Mock()
    mock_chat = Mock()
    mock_completions = Mock()
    mock_completions.create.return_value = mock_openai_response
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat
    return mock_client


@patch("arize_experiment.evaluators.sentiment_classification_accuracy.OpenAI")
def test_evaluator_initialization(mock_openai: Mock) -> None:
    """Test that the evaluator initializes with default and custom parameters."""
    # Test with defaults
    evaluator = SentimentClassificationAccuracyEvaluator()
    assert evaluator.name == "sentiment_classification_accuracy"
    assert evaluator._model == "gpt-4o-mini"
    assert evaluator._temperature == 0

    # Test with custom parameters
    evaluator = SentimentClassificationAccuracyEvaluator(
        model="gpt-4", temperature=0.7, api_key="test-key"
    )
    assert evaluator._model == "gpt-4"
    assert evaluator._temperature == 0.7


@patch("arize_experiment.evaluators.sentiment_classification_accuracy.OpenAI")
def test_parse_llm_output(mock_openai: Mock) -> None:
    """Test the LLM output parsing."""
    evaluator = SentimentClassificationAccuracyEvaluator()

    # Test correct case
    correct, explanation = evaluator._parse_llm_output(
        "correct The sentiment is accurate because..."
    )
    assert correct is True
    assert "the sentiment is accurate because" in explanation.lower()

    # Test incorrect case
    correct, explanation = evaluator._parse_llm_output(
        "incorrect The sentiment is wrong because..."
    )
    assert correct is False
    assert "the sentiment is wrong because" in explanation.lower()

    # Test with different casing and whitespace
    correct, explanation = evaluator._parse_llm_output("  CORRECT   The explanation...")
    assert correct is True
    assert "the explanation" in explanation.lower()

    # Test invalid format
    with pytest.raises(ValueError):
        evaluator._parse_llm_output("invalid format")


@patch("arize_experiment.evaluators.sentiment_classification_accuracy.OpenAI")
def test_evaluate_success(mock_openai: Mock) -> None:
    """Test successful evaluation."""
    # Create a properly structured mock response
    mock_message = Mock()
    mock_message.content = "correct This sentiment classification is accurate"

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_completion = Mock()
    mock_completion.choices = [mock_choice]

    mock_chat = Mock()
    mock_chat.completions.create.return_value = mock_completion

    mock_client = Mock()
    mock_client.chat = mock_chat

    mock_openai.return_value = mock_client

    evaluator = SentimentClassificationAccuracyEvaluator()
    task_result = TaskResult(
        dataset_row={"input": "This is a great product!"},
        output="positive",
        metadata={},
    )

    result = evaluator.evaluate(task_result)

    assert isinstance(result, EvaluationResult)
    assert result.score == 1.0
    assert result.label == "correct"
    assert result.explanation is not None


@patch("arize_experiment.evaluators.sentiment_classification_accuracy.OpenAI")
def test_evaluate_api_error(mock_openai: Mock) -> None:
    """Test evaluation when API call fails."""
    # Create a mock client that raises an exception
    error_client = Mock()
    error_chat = Mock()
    error_completions = Mock()
    error_completions.create.side_effect = Exception("API Error")
    error_chat.completions = error_completions
    error_client.chat = error_chat

    mock_openai.return_value = error_client

    evaluator = SentimentClassificationAccuracyEvaluator()
    task_result = TaskResult(
        dataset_row={"input": "This is a great product!"},
        output="positive",
        metadata={},
    )

    result = evaluator.evaluate(task_result)
    assert isinstance(result, EvaluationResult)
    assert result.score == 0.0
    assert result.label == "error"
    assert "API Error" in result.explanation


@patch("arize_experiment.evaluators.sentiment_classification_accuracy.OpenAI")
def test_callable_interface(mock_openai: Mock) -> None:
    """Test that the evaluator works as a callable."""
    evaluator = SentimentClassificationAccuracyEvaluator()

    # Create a properly structured mock response
    mock_message = Mock()
    mock_message.content = "correct Good classification"

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_completion = Mock()
    mock_completion.choices = [mock_choice]

    mock_chat = Mock()
    mock_chat.completions.create.return_value = mock_completion

    mock_client = Mock()
    mock_client.chat = mock_chat

    evaluator._client = mock_client

    # Test with TaskResult
    task_result = TaskResult(
        dataset_row={"input": "This is great!"},
        output="positive",
        metadata={},
    )
    result = evaluator(task_result)
    assert isinstance(result, EvaluationResult)
    assert result.score == 1.0
    assert result.label == "correct"

    # Test with dictionary
    dict_result = {
        "input": {"input": "This is great!"},
        "output": "positive",
        "metadata": {},
    }
    result = evaluator(dict_result)
    assert isinstance(result, EvaluationResult)
    assert result.score == 1.0
    assert result.label == "correct"

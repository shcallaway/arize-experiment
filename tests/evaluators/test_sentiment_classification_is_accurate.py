"""Test sentiment classification accuracy evaluator."""

from unittest.mock import MagicMock, patch

import pytest
from arize.experimental.datasets.experiments.types import EvaluationResult
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)

from arize_experiment.core.task import TaskResult
from arize_experiment.evaluators.sentiment_classification_is_accurate import (
    SentimentClassificationAccuracyEvaluator,
)


@pytest.fixture
def mock_chat_completion() -> ChatCompletion:
    """Create a mock chat completion."""
    return ChatCompletion(
        id="test",
        model="test",
        object="chat.completion",
        created=123,
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="correct - The sentiment matches the text.",
                ),
            )
        ],
    )


@pytest.fixture
def mock_openai(mock_chat_completion: ChatCompletion) -> MagicMock:
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    return mock_client


@patch("arize_experiment.evaluators.sentiment_classification_is_accurate.OpenAI")
def test_evaluator_name(mock_openai_class: MagicMock) -> None:
    """Test that the evaluator name is correct."""
    evaluator = SentimentClassificationAccuracyEvaluator()
    assert evaluator.name == "sentiment_classification_is_accurate"


def test_invalid_threshold() -> None:
    """Test that invalid thresholds raise ValueError."""
    with pytest.raises(ValueError):
        SentimentClassificationAccuracyEvaluator(threshold=1.5)
    with pytest.raises(ValueError):
        SentimentClassificationAccuracyEvaluator(threshold=-0.1)


@patch("arize_experiment.evaluators.sentiment_classification_is_accurate.OpenAI")
def test_evaluate_correct_sentiment(
    mock_openai_class: MagicMock, mock_openai: MagicMock
) -> None:
    """Test evaluation of correct sentiment."""
    mock_openai_class.return_value = mock_openai

    evaluator = SentimentClassificationAccuracyEvaluator()
    result = evaluator.evaluate(
        TaskResult(
            dataset_row={"input": "Great product!"},
            output="positive",
            metadata={},
        )
    )

    assert isinstance(result, EvaluationResult)
    assert result.score == 1.0
    assert result.label == "correct"
    assert "matches" in result.explanation.lower()


@patch("arize_experiment.evaluators.sentiment_classification_is_accurate.OpenAI")
def test_evaluate_incorrect_sentiment(
    mock_openai_class: MagicMock,
    mock_openai: MagicMock,
    mock_chat_completion: ChatCompletion,
) -> None:
    """Test evaluation of incorrect sentiment."""
    mock_chat_completion.choices[0].message.content = (
        "incorrect - The sentiment is wrong."
    )
    mock_openai.chat.completions.create.return_value = mock_chat_completion
    mock_openai_class.return_value = mock_openai

    evaluator = SentimentClassificationAccuracyEvaluator()
    result = evaluator.evaluate(
        TaskResult(
            dataset_row={"input": "Terrible service!"},
            output="positive",
            metadata={},
        )
    )

    assert isinstance(result, EvaluationResult)
    assert result.score == 0.0
    assert result.label == "incorrect"
    assert "wrong" in result.explanation.lower()


@patch("arize_experiment.evaluators.sentiment_classification_is_accurate.OpenAI")
def test_evaluate_invalid_input(mock_openai_class: MagicMock) -> None:
    """Test evaluation with invalid input."""
    evaluator = SentimentClassificationAccuracyEvaluator()
    result = evaluator.evaluate(
        TaskResult(
            dataset_row={"input": None},
            output="positive",
            metadata={},
        )
    )

    assert isinstance(result, EvaluationResult)
    assert result.score == 0.0
    assert result.label == "error"
    assert "failed" in result.explanation.lower()


@patch("arize_experiment.evaluators.sentiment_classification_is_accurate.OpenAI")
def test_evaluate_empty_llm_response(
    mock_openai_class: MagicMock,
    mock_openai: MagicMock,
    mock_chat_completion: ChatCompletion,
) -> None:
    """Test evaluation with empty LLM response."""
    mock_chat_completion.choices[0].message.content = None
    mock_openai.chat.completions.create.return_value = mock_chat_completion
    mock_openai_class.return_value = mock_openai

    evaluator = SentimentClassificationAccuracyEvaluator()
    result = evaluator.evaluate(
        TaskResult(
            dataset_row={"input": "Great product!"},
            output="positive",
            metadata={},
        )
    )

    assert isinstance(result, EvaluationResult)
    assert result.score == 0.0
    assert result.label == "error"
    assert "empty response" in result.explanation.lower()

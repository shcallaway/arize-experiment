"""
Tests for the agent response quality evaluator.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from arize.experimental.datasets.experiments.types import EvaluationResult
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from arize_experiment.core.task import TaskResult
from arize_experiment.evaluators.agent_response_quality import (
    AgentResponseQualityEvaluator,
)


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = MagicMock(spec=ChatCompletion)
    mock_message = MagicMock(spec=ChatCompletionMessage)
    mock_message.content = (
        "Score: 0.85\n"
        "The response is clear, relevant, and "
        "addresses all aspects of the query."
    )
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


@pytest.fixture(autouse=True)
def mock_openai_client():
    """Create a mock OpenAI client."""
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}),
        patch("openai.OpenAI") as mock,
    ):
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


def test_evaluator_initialization(mock_openai_client):
    """Test evaluator initialization with default parameters."""
    evaluator = AgentResponseQualityEvaluator()
    assert evaluator.name == "agent_response_quality"
    assert evaluator._model == "gpt-4o-mini"
    assert evaluator._temperature == 0


def test_evaluator_initialization_with_params(mock_openai_client):
    """Test evaluator initialization with custom parameters."""
    evaluator = AgentResponseQualityEvaluator(
        model="gpt-3.5-turbo", temperature=0.5, api_key="test-key"
    )
    assert evaluator._model == "gpt-3.5-turbo"
    assert evaluator._temperature == 0.5


def test_parse_llm_output_valid(mock_openai_client):
    """Test parsing valid LLM output."""
    evaluator = AgentResponseQualityEvaluator()
    output = "Score: 0.85\nExplanation: The response is clear and relevant."
    score, explanation = evaluator._parse_llm_output(output)
    assert score == 0.85
    assert explanation == "The response is clear and relevant."


def test_parse_llm_output_invalid(mock_openai_client):
    """Test parsing invalid LLM output."""
    evaluator = AgentResponseQualityEvaluator()
    invalid_outputs = [
        "Invalid format",
        "Score: invalid\nExplanation: test",
        "Score: 1.5\nExplanation: test",
        "Score: 0.8",
        "\nExplanation: test",
    ]
    for output in invalid_outputs:
        with pytest.raises(ValueError):
            evaluator._parse_llm_output(output)


@patch("openai.OpenAI")
def test_evaluate_success(mock_openai_client, mock_openai_response):
    """Test successful evaluation of agent response."""
    # Create a mock response with the expected content
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    "Score: 0.85\n"
                    "The response is clear, relevant, and "
                    "addresses all aspects of the query."
                )
            )
        )
    ]

    # Setup mock client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_client.return_value = mock_client

    # Create evaluator with mocked client
    evaluator = AgentResponseQualityEvaluator()
    evaluator._client = mock_client  # Directly inject the mock client

    # Create test task result
    task_result = TaskResult(
        dataset_row={
            "input": json.dumps(
                [
                    "What is Python?",
                    "Python is a high-level programming language.",
                    "Can you give an example?",
                ]
            )
        },
        output={
            "response": (
                "Here's a simple example of Python code:\n\n"
                "print('Hello, World!')\n\n"
                "This code prints the text 'Hello, World!' to the console."
            )
        },
        metadata={},
    )

    # Evaluate
    result = evaluator.evaluate(task_result)

    # Verify result
    assert isinstance(result, EvaluationResult)
    assert result.score == 0.85
    assert result.label == "good"
    assert (
        result.explanation
        == "The response is clear, relevant, and addresses all aspects of the query."
    )

    # Verify API call
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0
    assert len(call_kwargs["messages"]) == 2


def test_evaluate_missing_data(mock_openai_client):
    """Test evaluation with missing conversation or response."""
    evaluator = AgentResponseQualityEvaluator()

    # Test missing conversation (causes JSON decode error)
    task_result = TaskResult(
        dataset_row={},
        output={"response": "Test response"},
        metadata={},
    )
    with pytest.raises(json.JSONDecodeError):
        evaluator.evaluate(task_result)

    # Test missing response
    task_result = TaskResult(
        dataset_row={"input": json.dumps(["Test message"])},
        output={},
        metadata={},
    )
    with pytest.raises(ValueError, match="Missing agent response"):
        evaluator.evaluate(task_result)


def test_call_with_dict(mock_openai_client):
    """Test calling evaluator with dictionary input."""
    evaluator = AgentResponseQualityEvaluator()

    with patch.object(evaluator, "evaluate") as mock_evaluate:
        mock_evaluate.return_value = EvaluationResult(
            score=0.85, label="good", explanation="Test explanation"
        )

        result = evaluator(
            {
                "dataset_row": {"conversation": ["Test message"]},
                "output": "Test response",
                "metadata": {},
            }
        )

        assert isinstance(result, EvaluationResult)
        assert result.score == 0.85
        assert result.label == "good"
        assert result.explanation == "Test explanation"

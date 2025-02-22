"""
Tests for the chatbot response alignment evaluator.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from arize.experimental.datasets.experiments.types import EvaluationResult
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from arize_experiment.core.task import TaskResult
from arize_experiment.evaluators.chatbot_response_is_aligned import (
    ChatbotResponseIsAlignedEvaluator,
)


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = MagicMock(spec=ChatCompletion)
    mock_message = MagicMock(spec=ChatCompletionMessage)
    mock_message.parsed = {
        "score": True,
        "explanation": "The response aligns with the expected script."
    }
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
    evaluator = ChatbotResponseIsAlignedEvaluator(agent_id="test-agent")
    assert evaluator.name == "chatbot_response_is_aligned"
    assert evaluator._model == "gpt-4o-mini"
    assert evaluator._temperature == 0


def test_evaluator_initialization_with_params(mock_openai_client):
    """Test evaluator initialization with custom parameters."""
    evaluator = ChatbotResponseIsAlignedEvaluator(
        agent_id="test-agent",
        model="gpt-3.5-turbo",
        temperature=0.5,
        api_key="test-key"
    )
    assert evaluator._model == "gpt-3.5-turbo"
    assert evaluator._temperature == 0.5


def test_parse_llm_output_valid(mock_openai_client):
    """Test parsing valid LLM output."""
    evaluator = ChatbotResponseIsAlignedEvaluator(agent_id="test-agent")
    output = {"score": True, "explanation": "The response aligns with the expected script."}
    score, explanation = evaluator._parse_llm_output(output)
    assert score is True
    assert explanation == "The response aligns with the expected script."


def test_parse_llm_output_invalid(mock_openai_client):
    """Test parsing invalid LLM output."""
    evaluator = ChatbotResponseIsAlignedEvaluator(agent_id="test-agent")
    invalid_outputs = [
        {"score": "invalid", "explanation": "test"},
        {"score": 1.5, "explanation": "test"},
        {"score": True},
        {"explanation": "test"},
    ]
    for output in invalid_outputs:
        assert isinstance(output, dict) 
        with pytest.raises(ValueError):
            evaluator._parse_llm_output(output)


@patch("openai.OpenAI")
def test_evaluate_success(mock_openai_client, mock_openai_response):
    """Test successful evaluation of chatbot response alignment."""
    # Setup mock client
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = mock_openai_response
    mock_openai_client.return_value = mock_client

    # Create evaluator with mocked client
    evaluator = ChatbotResponseIsAlignedEvaluator(agent_id="test-agent")
    evaluator._client = mock_client 

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
    assert result.score is True
    assert result.label == "acceptable"
    assert result.explanation == "The response aligns with the expected script."

    # Verify API call
    mock_client.chat.completions.parse.assert_called_once()
    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0
    assert len(call_kwargs["messages"]) == 2


def test_evaluate_missing_data(mock_openai_client):
    """Test evaluation with missing conversation or response."""
    evaluator = ChatbotResponseIsAlignedEvaluator(agent_id="test-agent")

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
    with pytest.raises(ValueError, match="Missing chatbot response"):
        evaluator.evaluate(task_result)


def test_call_with_dict(mock_openai_client):
    """Test calling evaluator with dictionary input."""
    evaluator = ChatbotResponseIsAlignedEvaluator(agent_id="test-agent")

    with patch.object(evaluator, "evaluate") as mock_evaluate:
        mock_evaluate.return_value = EvaluationResult(
            score=True, label="acceptable", explanation="Test explanation"
        )

        result = evaluator(
            {
                "dataset_row": {"conversation": ["Test message"]},
                "output": "Test response",
                "metadata": {},
            }
        )

        assert isinstance(result, EvaluationResult)
        assert result.score is True
        assert result.label == "acceptable"
        assert result.explanation == "Test explanation"

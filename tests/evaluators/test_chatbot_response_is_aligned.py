"""
Tests for the chatbot response alignment evaluator.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from arize.experimental.datasets.experiments.types import EvaluationResult
from openai.types.chat import ChatCompletion

from arize_experiment.core.task import TaskResult
from arize_experiment.evaluators.chatbot_response_is_acceptable.evaluator import ChatbotResponseIsAcceptableEvaluator


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = MagicMock(spec=ChatCompletion)
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "relevance": 0.9,
        "accuracy": 0.8,
        "clarity": 0.7,
        "completeness": 0.8,
        "appropriateness": 0.9,
        "explanation": "The response is relevant, accurate, clear, complete, and appropriate."
    })
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
    evaluator = ChatbotResponseIsAcceptableEvaluator()
    assert evaluator.name == "chatbot_response_is_acceptable"
    assert evaluator._model == "gpt-4o-mini"
    assert evaluator._temperature == 0


def test_evaluator_initialization_with_params(mock_openai_client):
    """Test evaluator initialization with custom parameters."""
    evaluator = ChatbotResponseIsAcceptableEvaluator(
        model="gpt-3.5-turbo",
        temperature=0.5,
        api_key="test-key"
    )
    assert evaluator._model == "gpt-3.5-turbo"
    assert evaluator._temperature == 0.5


def test_parse_llm_output_valid(mock_openai_client):
    """Test parsing valid LLM output."""
    evaluator = ChatbotResponseIsAcceptableEvaluator()
    output = {
        "relevance": 0.9,
        "accuracy": 0.8,
        "clarity": 0.7,
        "completeness": 0.8,
        "appropriateness": 0.9,
        "explanation": "The response is relevant, accurate, clear, complete, and appropriate."
    }
    score, explanation = evaluator._parse_llm_output(output)
    assert score == 0.82  # (0.9 + 0.8 + 0.7 + 0.8 + 0.9) / 5
    assert explanation == "The response is relevant, accurate, clear, complete, and appropriate."


def test_parse_llm_output_invalid(mock_openai_client):
    """Test parsing invalid LLM output."""
    evaluator = ChatbotResponseIsAcceptableEvaluator()
    invalid_outputs = [
        {"relevance": "invalid", "accuracy": 0.8, "clarity": 0.7, "completeness": 0.8, "appropriateness": 0.9, "explanation": "test"},
        {"relevance": 0.9, "accuracy": 0.8, "clarity": 0.7, "completeness": 0.8, "explanation": "test"},
        {"relevance": 0.9, "accuracy": 0.8, "clarity": 0.7, "appropriateness": 0.9, "explanation": "test"},
        {},
    ]
    for output in invalid_outputs:
        assert isinstance(output, dict) 
        with pytest.raises(ValueError):
            evaluator._parse_llm_output(output)


@patch("openai.OpenAI")
def test_evaluate_success(mock_openai_class, mock_openai_response):
    """Test successful evaluation of chatbot response quality."""
    # Setup mock client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    mock_openai_class.return_value = mock_client

    # Create evaluator with mocked client
    evaluator = ChatbotResponseIsAcceptableEvaluator()
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
    assert result.score == 0.82  # (0.9 + 0.8 + 0.7 + 0.8 + 0.9) / 5
    assert result.label == "good"
    assert result.explanation == "The response is relevant, accurate, clear, complete, and appropriate."

    # Verify API call
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0
    assert len(call_kwargs["messages"]) == 2


def test_evaluate_missing_data(mock_openai_client):
    """Test evaluation with missing conversation or response."""
    evaluator = ChatbotResponseIsAcceptableEvaluator()

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


def test_determine_label():
    """Test the label determination based on score."""
    evaluator = ChatbotResponseIsAcceptableEvaluator()
    
    assert evaluator._determine_label(0.95) == "excellent"
    assert evaluator._determine_label(0.9) == "excellent"
    assert evaluator._determine_label(0.8) == "good"
    assert evaluator._determine_label(0.7) == "good"
    assert evaluator._determine_label(0.6) == "fair"
    assert evaluator._determine_label(0.5) == "fair"
    assert evaluator._determine_label(0.4) == "poor"
    assert evaluator._determine_label(0.3) == "poor"
    assert evaluator._determine_label(0.2) == "unacceptable"
    assert evaluator._determine_label(0.0) == "unacceptable"


def test_format_conversation():
    """Test the conversation formatting."""
    evaluator = ChatbotResponseIsAcceptableEvaluator()
    
    conversation = ["Hello", "Hi there", "How are you?", "I'm good, thanks!"]
    formatted = evaluator._format_conversation(conversation)
    
    expected = "User: Hello\nAssistant: Hi there\nUser: How are you?\nAssistant: I'm good, thanks!"
    assert formatted == expected

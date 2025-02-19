"""
Tests for the sentiment classification task.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any
from dotenv import load_dotenv

from arize_experiment.tasks.sentiment_classification import SentimentClassificationTask
from arize_experiment.core.errors import TaskError

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_message = Mock()
    mock_message.content = "positive"
    
    mock_choice = Mock()
    mock_choice.message = mock_message
    
    mock_completion = Mock()
    mock_completion.choices = [mock_choice]
    
    return mock_completion


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Create a mock OpenAI client."""
    mock_client = Mock()
    mock_chat = Mock()
    mock_completions = Mock()
    mock_completions.create.return_value = mock_openai_response
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat
    return mock_client


def test_task_initialization():
    """Test that the task initializes with default and custom parameters."""
    # Test with defaults
    task = SentimentClassificationTask()
    assert task.name == "sentiment_classification"
    assert task._model == "gpt-4o-mini"
    assert task._temperature == 0
    
    # Test with custom parameters
    task = SentimentClassificationTask(
        model="gpt-4",
        temperature=0.7,
        api_key="test-key"
    )
    assert task._model == "gpt-4"
    assert task._temperature == 0.7


@patch('arize_experiment.tasks.sentiment_classification.OpenAI')
def test_execute_valid_input(mock_openai, mock_openai_client):
    """Test task execution with valid input."""
    mock_openai.return_value = mock_openai_client
    
    task = SentimentClassificationTask()
    result = task.execute({"input": "This is a great day!"})
    
    assert result.success
    assert result.output == "positive"
    assert result.metadata == {
        "model": "gpt-4o-mini",
        "temperature": 0
    }
    assert result.error is None


@patch('arize_experiment.tasks.sentiment_classification.OpenAI')
def test_execute_invalid_input_format(mock_openai, mock_openai_client):
    """Test task execution with invalid input format."""
    mock_openai.return_value = mock_openai_client
    
    task = SentimentClassificationTask()
    
    # Test with non-dict input
    result = task.execute("invalid input")
    assert not result.success
    assert result.error is not None
    assert "must be a dictionary" in result.error.lower()
    
    # Test with dict missing 'input' key
    result = task.execute({"wrong_key": "text"})
    assert not result.success
    assert result.error is not None
    assert "must be a dictionary with 'input' key" in result.error.lower()
    
    # Test with non-string input
    result = task.execute({"input": 123})
    assert not result.success
    assert result.error is not None
    assert "must be a string" in result.error.lower()


@patch('arize_experiment.tasks.sentiment_classification.OpenAI')
def test_execute_api_error(mock_openai):
    """Test task execution when API call fails."""
    # Create a new mock client that raises an exception
    error_client = Mock()
    error_chat = Mock()
    error_completions = Mock()
    error_completions.create.side_effect = Exception("API Error")
    error_chat.completions = error_completions
    error_client.chat = error_chat
    
    mock_openai.return_value = error_client
    
    task = SentimentClassificationTask()
    result = task.execute({"input": "test text"})
    
    assert not result.success
    assert result.error is not None
    assert "api error" in result.error.lower()


def test_parse_llm_output():
    """Test LLM output parsing."""
    task = SentimentClassificationTask()
    
    # Test valid outputs
    assert task._parse_llm_output("positive") == "positive"
    assert task._parse_llm_output("NEGATIVE") == "negative"
    assert task._parse_llm_output(" neutral ") == "neutral"
    
    # Test with whitespace
    assert task._parse_llm_output("  positive\n") == "positive"


@patch('arize_experiment.tasks.sentiment_classification.OpenAI')
def test_execute_end_to_end(mock_openai):
    """Test complete end-to-end task execution."""
    # Create a fresh mock for this test
    mock_message = Mock()
    mock_message.content = "positive"
    
    mock_choice = Mock()
    mock_choice.message = mock_message
    
    mock_completion = Mock()
    mock_completion.choices = [mock_choice]
    
    mock_completions = Mock()
    mock_completions.create.return_value = mock_completion
    
    mock_chat = Mock()
    mock_chat.completions = mock_completions
    
    mock_client = Mock()
    mock_client.chat = mock_chat
    
    mock_openai.return_value = mock_client
    
    task = SentimentClassificationTask(
        model="gpt-4",
        temperature=0.5
    )
    
    result = task.execute({"input": "I love this product!"})
    
    # Verify the API was called with correct parameters
    mock_completions.create.assert_called_once()
    call_args = mock_completions.create.call_args[1]
    assert call_args["model"] == "gpt-4"
    assert call_args["temperature"] == 0.5
    assert len(call_args["messages"]) == 2
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][1]["role"] == "user"
    assert call_args["messages"][1]["content"] == "I love this product!"
    
    # Verify the result
    assert result.success
    assert result.output == "positive"
    assert result.metadata["model"] == "gpt-4"
    assert result.metadata["temperature"] == 0.5 

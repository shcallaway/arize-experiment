"""Tests for the execute agent task."""

from typing import Any, Dict, cast
from unittest.mock import Mock, patch

import pytest
import requests
from requests.exceptions import RequestException

from arize_experiment.tasks.execute_agent import ExecuteAgentTask


def test_task_initialization() -> None:
    """Test that the task initializes with the correct URL."""
    url = "http://localhost:8000/agent"
    task = ExecuteAgentTask(url=url)
    assert task.url == url
    assert task.name == "execute_agent"


def test_task_name() -> None:
    """Test the task name property."""
    task = ExecuteAgentTask(url="http://test.com")
    assert task.name == "execute_agent"


def test_default_url() -> None:
    """Test that the default URL is set correctly."""
    task = ExecuteAgentTask()
    assert task.url == "http://localhost:8080"


@pytest.mark.parametrize(
    "input_data",
    [
        "not a dict",
        123,
        None,
        [],
        {},  # Missing required fields
        {"agent_id": "123"},  # Missing conversation
        {"conversation": []},  # Missing agent_id
        {"agent_id": "123", "conversation": "not a list"},  # Invalid conversation type
        {"agent_id": "123", "conversation": [123]},  # Invalid message type
        {
            "agent_id": "123",
            "conversation": [{"role": "user"}],  # Missing content
        },
        {
            "agent_id": "123",
            "conversation": [{"content": "hello"}],  # Missing role
        },
    ],
)
def test_execute_invalid_input_format(input_data: object) -> None:
    """Test task execution with invalid input formats."""
    task = ExecuteAgentTask(url="http://test.com")
    result = task.execute(input_data)

    assert not result.success
    assert result.error is not None
    assert result.output is None
    assert result.metadata == {"url": task.url}


@patch("requests.post")
def test_successful_request(mock_post: Mock) -> None:
    """Test successful API request and response handling."""
    # Setup mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Hello, I can help with that!"}
    mock_response.headers = {"Content-Type": "application/json"}
    mock_post.return_value = mock_response

    task = ExecuteAgentTask()
    input_data = {
        "agent_id": "123",
        "conversation": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ],
    }

    result = task.execute(input_data)

    # Verify the request was made correctly
    mock_post.assert_called_once_with(task.url, json=input_data)

    # Verify the response was processed correctly
    assert result.success
    assert result.error is None
    assert result.output == {"response": "Hello, I can help with that!"}

    # Cast metadata to Dict[str, Any] since we know it exists in this success case
    metadata = cast(Dict[str, Any], result.metadata)
    assert metadata["url"] == task.url
    assert metadata["status_code"] == 200
    assert metadata["headers"] == {"Content-Type": "application/json"}


@patch("requests.post")
def test_request_error_handling(mock_post: Mock) -> None:
    """Test handling of various request errors."""
    # Setup mock to raise an exception
    mock_post.side_effect = RequestException("Connection error")

    task = ExecuteAgentTask()
    input_data = {
        "agent_id": "123",
        "conversation": [{"role": "user", "content": "Hello"}],
    }

    result = task.execute(input_data)

    # Verify error handling
    assert not result.success
    assert result.output is None
    assert "Failed to execute agent request" in str(result.error)
    assert result.metadata == {"url": task.url}


@patch("requests.post")
def test_http_error_handling(mock_post: Mock) -> None:
    """Test handling of HTTP error responses."""
    # Setup mock to return a 404 error
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")
    mock_post.return_value = mock_response

    task = ExecuteAgentTask()
    input_data = {
        "agent_id": "123",
        "conversation": [{"role": "user", "content": "Hello"}],
    }

    result = task.execute(input_data)

    # Verify error handling
    assert not result.success
    assert result.output is None
    assert "Failed to execute agent request" in str(result.error)
    assert "404 Client Error" in str(result.error)
    assert result.metadata == {"url": task.url}


@patch("requests.post")
def test_json_decode_error(mock_post: Mock) -> None:
    """Test handling of invalid JSON response."""
    # Setup mock to return invalid JSON
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_post.return_value = mock_response

    task = ExecuteAgentTask()
    input_data = {
        "agent_id": "123",
        "conversation": [{"role": "user", "content": "Hello"}],
    }

    result = task.execute(input_data)

    # Verify error handling
    assert not result.success
    assert result.output is None
    assert "Agent execution failed" in str(result.error)
    assert "Invalid JSON" in str(result.error)
    assert result.metadata == {"url": task.url}

"""Tests for the chatbot server task."""

import json
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
import requests
from requests.exceptions import RequestException

from arize_experiment.tasks.call_chatbot_server import CallChatbotServerTask


def test_task_initialization() -> None:
    """Test that the task initializes with the correct URL."""
    base_url = "http://localhost:8000"
    task = CallChatbotServerTask(base_url=base_url)
    assert task.url == f"{base_url}/execute"
    assert task.name == "call_chatbot_server"


def test_task_name() -> None:
    """Test the task name property."""
    task = CallChatbotServerTask()
    assert task.name == "call_chatbot_server"


def test_default_url() -> None:
    """Test that the default URL is set correctly."""
    task = CallChatbotServerTask()
    assert task.url == "http://localhost:8080/execute"


@pytest.mark.parametrize(
    "input_data",
    [
        {"input": "not_json"},  # Invalid JSON string
        {"input": "123"},  # JSON but not a list
        {"input": "[]"},  # Empty list
        {"wrong_key": "[]"},  # Missing input key
    ],
)
def test_execute_invalid_input_format(input_data: Dict[str, Any]) -> None:
    """Test task execution with invalid input formats."""
    task = CallChatbotServerTask()
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

    task = CallChatbotServerTask()
    conversation = [{"role": "user", "content": "What is the weather in Tokyo?"}]
    input_data = {"input": json.dumps(conversation)}

    expected_payload = {"agent_id": "test", "conversation": conversation}

    result = task.execute(input_data)

    # Verify the request was made correctly
    mock_post.assert_called_once_with(task.url, json=expected_payload)

    # Verify the response was processed correctly
    assert result.success
    assert result.output is not None and result.output == {
        "response": "Hello, I can help with that!"
    }
    assert result.metadata is not None and result.metadata["status_code"] == 200
    assert result.metadata is not None and result.metadata["headers"] == {
        "Content-Type": "application/json"
    }


@patch("requests.post")
def test_request_error_handling(mock_post: Mock) -> None:
    """Test handling of various request errors."""
    # Setup mock to raise an exception
    mock_post.side_effect = RequestException("Connection error")

    task = CallChatbotServerTask()
    conversation = [{"role": "user", "content": "Hello"}]
    input_data = {"input": json.dumps(conversation)}

    result = task.execute(input_data)

    # Verify error handling
    assert not result.success
    assert result.output is None
    assert "Failed to execute agent request" in str(result.error)


@patch("requests.post")
def test_http_error_handling(mock_post: Mock) -> None:
    """Test handling of HTTP error responses."""
    # Setup mock to return a 404 error
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")
    mock_post.return_value = mock_response

    task = CallChatbotServerTask()
    conversation = [{"role": "user", "content": "Hello"}]
    input_data = {"input": json.dumps(conversation)}

    result = task.execute(input_data)

    # Verify error handling
    assert not result.success
    assert result.output is None
    assert "Failed to execute agent request" in str(result.error)


@patch("requests.post")
def test_json_decode_error(mock_post: Mock) -> None:
    """Test handling of invalid JSON response."""
    # Setup mock to return invalid JSON
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_post.return_value = mock_response

    task = CallChatbotServerTask()
    conversation = [{"role": "user", "content": "Hello"}]
    input_data = {"input": json.dumps(conversation)}

    result = task.execute(input_data)

    # Verify error handling
    assert not result.success
    assert result.output is None
    assert "Invalid JSON response" in str(result.error)

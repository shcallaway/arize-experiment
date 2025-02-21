"""Tests for the delegate task."""

import json
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
import requests
from requests.exceptions import RequestException

from arize_experiment.tasks.delegate import DelegateTask


def test_task_initialization() -> None:
    """Test that the task initializes with the correct URL."""
    base_url = "http://localhost:8000"
    task = DelegateTask(base_url=base_url)
    assert task.url == f"{base_url}/delegate"
    assert task.name == "delegate"


def test_task_name() -> None:
    """Test the task name property."""
    task = DelegateTask()
    assert task.name == "delegate"


def test_default_url() -> None:
    """Test that the default URL is set correctly."""
    task = DelegateTask()
    assert task.url == "http://localhost:8080/delegate"


@pytest.mark.parametrize(
    "input_data",
    [
        {"input": "not_json"},  # Invalid JSON string
        {"input": "[]"},  # JSON array instead of object
        {"input": "123"},  # JSON number instead of object
        {"wrong_key": "{}"},  # Missing input key
    ],
)
def test_execute_invalid_input_format(input_data: Dict[str, Any]) -> None:
    """Test task execution with invalid input formats."""
    task = DelegateTask()
    result = task.execute(input_data)

    assert not result.success
    assert result.error is not None
    assert result.output is None
    assert result.metadata is not None
    assert result.metadata == {"url": task.url}


@patch("requests.post")
def test_successful_request(mock_post: Mock) -> None:
    """Test successful API request and response handling."""
    # Setup mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success", "result": "Task completed"}
    mock_response.headers = {"Content-Type": "application/json"}
    mock_post.return_value = mock_response

    task = DelegateTask()
    task_details = {"task_type": "process_data", "data": {"key": "value"}}
    input_data = {"input": json.dumps(task_details)}

    result = task.execute(input_data)

    # Verify the request was made correctly
    mock_post.assert_called_once_with(task.url, json=task_details)

    # Verify the response was processed correctly
    assert result.success
    assert result.output == {"status": "success", "result": "Task completed"}
    assert result.metadata is not None
    assert result.metadata["status_code"] == 200
    assert result.metadata["headers"] == {"Content-Type": "application/json"}


@patch("requests.post")
def test_request_error_handling(mock_post: Mock) -> None:
    """Test handling of various request errors."""
    # Setup mock to raise an exception
    mock_post.side_effect = RequestException("Connection error")

    task = DelegateTask()
    task_details = {"task_type": "process_data", "data": {"key": "value"}}
    input_data = {"input": json.dumps(task_details)}

    result = task.execute(input_data)

    # Verify error handling
    assert not result.success
    assert result.output is None
    assert "Failed to delegate task" in str(result.error)


@patch("requests.post")
def test_http_error_handling(mock_post: Mock) -> None:
    """Test handling of HTTP error responses."""
    # Setup mock to return a 404 error
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")
    mock_post.return_value = mock_response

    task = DelegateTask()
    task_details = {"task_type": "process_data", "data": {"key": "value"}}
    input_data = {"input": json.dumps(task_details)}

    result = task.execute(input_data)

    # Verify error handling
    assert not result.success
    assert result.output is None
    assert "Failed to delegate task" in str(result.error)


@patch("requests.post")
def test_json_decode_error(mock_post: Mock) -> None:
    """Test handling of invalid JSON response."""
    # Setup mock to return invalid JSON
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_post.return_value = mock_response

    task = DelegateTask()
    task_details = {"task_type": "process_data", "data": {"key": "value"}}
    input_data = {"input": json.dumps(task_details)}

    result = task.execute(input_data)

    # Verify error handling
    assert not result.success
    assert result.output is None
    assert "Invalid JSON response" in str(result.error)

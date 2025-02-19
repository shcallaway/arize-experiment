"""Tests for the execute agent task."""

import pytest

from arize_experiment.core.task import TaskResult
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


def test_execute_returns_task_result() -> None:
    """Test that execute returns a TaskResult with the expected structure."""
    url = "http://localhost:8000/agent"
    task = ExecuteAgentTask(url=url)
    input_data = {"prompt": "Hello agent"}

    result = task.execute(input_data)

    assert isinstance(result, TaskResult)
    assert result.input == input_data
    assert result.metadata == {"url": url}
    assert result.output is None  # Currently unimplemented as per the code


@pytest.mark.parametrize(
    "input_data",
    [
        "not a dict",
        123,
        None,
        [],
    ],
)
def test_execute_invalid_input_format(input_data: object) -> None:
    """Test task execution with invalid input formats."""
    task = ExecuteAgentTask(url="http://test.com")
    result = task.execute(input_data)

    assert not result.success
    assert result.error is not None
    assert "input must be a dictionary" in str(result.error).lower()

"""Tests for the task registry."""

import pytest

from arize_experiment.core.task import Task, TaskResult
from arize_experiment.core.task_registry import TaskRegistry


class MockTask(Task):
    """Mock task for testing."""

    @property
    def name(self) -> str:
        return "mock_task"

    def execute(self, Input: dict) -> TaskResult:
        return TaskResult(dataset_row=Input, output="mock_output")


def test_register_task() -> None:
    """Test registering a task."""
    # Clear registry before test
    TaskRegistry._tasks.clear()

    # Test registration via decorator
    @TaskRegistry.register("test_task")
    class TestTask(Task):
        @property
        def name(self) -> str:
            return "test_task"

        def execute(self, Input: dict) -> TaskResult:
            return TaskResult(dataset_row=Input, output="test_output")

    # Test direct registration
    TaskRegistry.register("mock_task", MockTask)

    # Verify registrations
    assert "test_task" in TaskRegistry._tasks
    assert "mock_task" in TaskRegistry._tasks
    assert len(TaskRegistry._tasks) == 2


def test_get_task() -> None:
    """Test getting a task by name."""
    # Clear registry before test
    TaskRegistry._tasks.clear()

    # Register a task
    TaskRegistry.register("mock_task", MockTask)

    # Test getting registered task
    task_class = TaskRegistry.get("mock_task")
    assert task_class == MockTask

    # Test getting non-existent task
    with pytest.raises(ValueError):
        TaskRegistry.get("nonexistent_task")


def test_list_tasks() -> None:
    """Test listing registered tasks."""
    # Clear registry before test
    TaskRegistry._tasks.clear()

    # Register tasks
    TaskRegistry.register("mock_task1", MockTask)
    TaskRegistry.register("mock_task2", MockTask)

    # Test listing tasks
    tasks = TaskRegistry.list()
    assert len(tasks) == 2
    assert "mock_task1" in tasks
    assert "mock_task2" in tasks


def test_duplicate_registration() -> None:
    """Test that registering a duplicate task name raises an error."""
    # Clear registry before test
    TaskRegistry._tasks.clear()

    # Register first task
    TaskRegistry.register("mock_task", MockTask)

    # Test registering duplicate
    with pytest.raises(ValueError):
        TaskRegistry.register("mock_task", MockTask)

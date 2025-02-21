"""Tests for the run command."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from arize_experiment.cli.commands.run import RunCommand
from arize_experiment.core.arize import ArizeDatasetProtocol
from arize_experiment.core.errors import ConfigurationError, HandlerError
from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.schema import DatasetSchema
from arize_experiment.core.task import Task, TaskResult
from arize_experiment.core.task_registry import TaskRegistry


@dataclass
class MockDataset(ArizeDatasetProtocol):
    """Mock dataset for testing."""

    data: List[Dict[str, Any]]

    def get_sample(self, limit: int) -> Any:
        """Get a sample of data."""
        return MagicMock(data=self.data[:limit])


class MockTask(Task):
    """Mock task for testing."""

    def __init__(self) -> None:
        """Initialize the mock task."""
        pass

    @property
    def name(self) -> str:
        """Get the task name."""
        return "mock_task"

    @property
    def required_schema(self) -> DatasetSchema:
        """Get the required schema."""
        return DatasetSchema(columns={})

    def execute(self, dataset_row: Dict[str, Any]) -> TaskResult:
        """Execute the task."""
        return TaskResult(
            dataset_row=dataset_row,
            output={"result": "mock_result"},
            metadata={"task": "mock_task"},
        )


class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing."""

    def __init__(self) -> None:
        """Initialize the mock evaluator."""
        pass

    @property
    def name(self) -> str:
        """Get the evaluator name."""
        return "mock_evaluator"

    def evaluate(self, task_result: TaskResult) -> Dict[str, Any]:
        """Evaluate the task result."""
        return {"score": 1.0}


@pytest.fixture
def mock_arize_client():
    """Create a mock Arize client."""
    client = MagicMock()
    client.get_dataset.return_value = MockDataset(
        data=[{"input": "test"}]
    )  # Dataset exists by default
    client.get_experiment.return_value = None  # Experiment doesn't exist by default
    client.run_experiment.return_value = {"success": True}
    return client


@pytest.fixture
def command(mock_arize_client):
    """Create a RunCommand instance with a mock Arize client."""
    with patch("arize_experiment.core.base_command.ArizeClient") as mock_client_class:
        mock_client_class.return_value = mock_arize_client
        with patch.dict(
            os.environ,
            {
                "ARIZE_API_KEY": "test_api_key",
                "ARIZE_SPACE_ID": "test_space_id",
                "ARIZE_DEVELOPER_KEY": "test_developer_key",
            },
        ):
            command = RunCommand()
            return command


@pytest.fixture(autouse=True)
def register_mock_task():
    """Fixture that registers the mock task."""
    TaskRegistry.register("mock_task", MockTask)
    yield
    TaskRegistry._tasks.pop("mock_task", None)


def test_run_success(command):
    """Test successful experiment run."""
    # Mock task and evaluator registries
    with patch("arize_experiment.cli.commands.run.TaskRegistry") as mock_task_registry:
        mock_task_registry.get.return_value = MockTask
        with patch(
            "arize_experiment.cli.commands.run.EvaluatorRegistry"
        ) as mock_evaluator_registry:
            mock_evaluator_registry.get.return_value = MockEvaluator

            # Run experiment
            command.execute(
                experiment_name="test_experiment",
                dataset_name="test_dataset",
                task_name="mock_task",
                evaluator_names=["mock_evaluator"],
            )

            # Verify the Arize client was called correctly
            command._arize_client.run_experiment.assert_called_once()
            call_args = command._arize_client.run_experiment.call_args[1]
            assert call_args["experiment_name"] == "test_experiment"
            assert call_args["dataset_name"] == "test_dataset"
            assert isinstance(call_args["task"], MockTask)
            assert len(call_args["evaluators"]) == 1
            assert isinstance(call_args["evaluators"][0], MockEvaluator)


def test_run_dataset_not_found(command):
    """Test error when dataset doesn't exist."""
    # Mock that the dataset doesn't exist
    command._arize_client.get_dataset.return_value = None

    # Verify that attempting to run raises an error
    with pytest.raises(ConfigurationError) as exc_info:
        command.execute(
            experiment_name="test_experiment",
            dataset_name="test_dataset",
            task_name="mock_task",
            evaluator_names=["mock_evaluator"],
        )
    assert "does not exist" in str(exc_info.value)


def test_run_experiment_exists(command):
    """Test error when experiment already exists."""
    # Mock that the experiment exists
    command._arize_client.get_experiment.return_value = {"id": "existing_id"}

    # Verify that attempting to run raises an error
    with pytest.raises(ConfigurationError) as exc_info:
        command.execute(
            experiment_name="test_experiment",
            dataset_name="test_dataset",
            task_name="mock_task",
            evaluator_names=["mock_evaluator"],
        )
    assert "already exists" in str(exc_info.value)


def test_run_invalid_task(command):
    """Test error when task is invalid."""
    # Mock task registry to raise an error
    with patch("arize_experiment.cli.commands.run.TaskRegistry") as mock_task_registry:
        mock_task_registry.get.side_effect = Exception("Invalid task")

        # Verify that attempting to run raises an error
        with pytest.raises(ConfigurationError) as exc_info:
            command.execute(
                experiment_name="test_experiment",
                dataset_name="test_dataset",
                task_name="invalid_task",
                evaluator_names=["mock_evaluator"],
            )
        assert "Failed to create task" in str(exc_info.value)


def test_run_no_evaluators(command):
    """Test error when no evaluators are provided."""
    # Verify that attempting to run raises an error
    with pytest.raises(HandlerError) as exc_info:
        command.execute(
            experiment_name="test_experiment",
            dataset_name="test_dataset",
            task_name="mock_task",
            evaluator_names=None,
        )
    assert "No evaluators provided" in str(exc_info.value)


def test_run_invalid_evaluator(command):
    """Test error when evaluator is invalid."""
    # Mock task registry
    with patch("arize_experiment.cli.commands.run.TaskRegistry") as mock_task_registry:
        mock_task_registry.get.return_value = MockTask
        # Mock evaluator registry to raise an error
        with patch(
            "arize_experiment.cli.commands.run.EvaluatorRegistry"
        ) as mock_evaluator_registry:
            mock_evaluator_registry.get.side_effect = Exception("Invalid evaluator")

            # Verify that attempting to run raises an error
            with pytest.raises(HandlerError) as exc_info:
                command.execute(
                    experiment_name="test_experiment",
                    dataset_name="test_dataset",
                    task_name="mock_task",
                    evaluator_names=["invalid_evaluator"],
                )
            assert "Failed to create evaluator" in str(exc_info.value)


def test_run_arize_error(command):
    """Test error when Arize client fails."""
    # Mock task and evaluator registries
    with patch("arize_experiment.cli.commands.run.TaskRegistry") as mock_task_registry:
        mock_task_registry.get.return_value = MockTask
        with patch(
            "arize_experiment.cli.commands.run.EvaluatorRegistry"
        ) as mock_evaluator_registry:
            mock_evaluator_registry.get.return_value = MockEvaluator

            # Mock Arize client to raise an error
            command._arize_client.run_experiment.side_effect = Exception(
                "Arize API error"
            )

            # Verify that attempting to run raises an error
            with pytest.raises(HandlerError) as exc_info:
                command.execute(
                    experiment_name="test_experiment",
                    dataset_name="test_dataset",
                    task_name="mock_task",
                    evaluator_names=["mock_evaluator"],
                )
            assert "Failed to run experiment" in str(exc_info.value)


def test_run_with_tags(command):
    """Test running experiment with tags."""
    # Mock task and evaluator registries
    with patch("arize_experiment.cli.commands.run.TaskRegistry") as mock_task_registry:
        mock_task_registry.get.return_value = MockTask
        with patch(
            "arize_experiment.cli.commands.run.EvaluatorRegistry"
        ) as mock_evaluator_registry:
            mock_evaluator_registry.get.return_value = MockEvaluator

            # Run experiment with tags
            command.execute(
                experiment_name="test_experiment",
                dataset_name="test_dataset",
                task_name="mock_task",
                evaluator_names=["mock_evaluator"],
                raw_tags=["key1=value1", "key2=value2"],
            )

            # Verify the Arize client was called correctly
            command._arize_client.run_experiment.assert_called_once()


def test_run_invalid_tag_format(command):
    """Test error when tag format is invalid."""
    # Verify that attempting to run raises an error
    with pytest.raises(ConfigurationError) as exc_info:
        command.execute(
            experiment_name="test_experiment",
            dataset_name="test_dataset",
            task_name="mock_task",
            evaluator_names=["mock_evaluator"],
            raw_tags=["invalid_tag"],
        )
    assert "Invalid tag format" in str(exc_info.value)

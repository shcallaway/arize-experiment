"""
Tests for CLI commands in arize-experiment.
"""

import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from arize_experiment.cli.cli import cli
from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize_experiment.core.schema import DatasetSchema
from arize_experiment.core.task import Task, TaskResult
from arize_experiment.core.task_registry import TaskRegistry


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
def cli_runner():
    """Fixture that provides a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_run_command():
    """Fixture that provides a mocked RunCommand instance."""
    with patch("arize_experiment.cli.cli.RunCommand") as mock:
        command_instance = MagicMock()
        mock.return_value = command_instance
        yield command_instance


@pytest.fixture
def mock_create_dataset_command():
    """Fixture that provides a mocked CreateDatasetCommand instance."""
    with patch("arize_experiment.cli.cli.CreateDatasetCommand") as mock:
        command_instance = MagicMock()
        mock.return_value = command_instance
        yield command_instance


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Fixture that mocks required environment variables."""
    with patch.dict(
        os.environ,
        {
            "ARIZE_API_KEY": "test-api-key",
            "ARIZE_SPACE_ID": "test-space-id",
            "ARIZE_DEVELOPER_KEY": "test-developer-key",
        },
    ):
        yield


@pytest.fixture(autouse=True)
def mock_registries():
    """Fixture that mocks the task and evaluator registries."""
    # Clear existing registrations
    TaskRegistry._tasks.clear()
    EvaluatorRegistry._evaluators.clear()

    # Register mock task and evaluator
    TaskRegistry.register("mock_task", MockTask)
    EvaluatorRegistry.register("mock_evaluator", MockEvaluator)

    # Patch the register function to prevent actual registrations
    with patch("arize_experiment.cli.cli.register"):
        yield

    # Clean up
    TaskRegistry._tasks.clear()
    EvaluatorRegistry._evaluators.clear()


def test_cli_help(cli_runner):
    """Test that the CLI help command works."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "arize-experiment: A tool for running experiments on Arize" in result.output


def test_run_command_minimal(cli_runner, mock_run_command):
    """Test the run command with minimal required arguments."""
    result = cli_runner.invoke(
        cli,
        [
            "run",
            "--name",
            "test-experiment",
            "--dataset",
            "test-dataset",
            "--task",
            "mock_task",
            "--evaluator",
            "mock_evaluator",
        ],
    )
    if result.exit_code != 0:
        print(f"\nCommand output:\n{result.output}")
    assert result.exit_code == 0
    mock_run_command.execute.assert_called_once_with(
        experiment_name="test-experiment",
        dataset_name="test-dataset",
        task_name="mock_task",
        raw_tags=None,
        evaluator_names=["mock_evaluator"],
    )


def test_run_command_with_tags(cli_runner, mock_run_command):
    """Test the run command with tags."""
    result = cli_runner.invoke(
        cli,
        [
            "run",
            "--name",
            "test-experiment",
            "--dataset",
            "test-dataset",
            "--task",
            "mock_task",
            "--evaluator",
            "mock_evaluator",
            "--tag",
            "env=test",
            "--tag",
            "version=1.0",
        ],
    )
    assert result.exit_code == 0
    mock_run_command.execute.assert_called_once_with(
        experiment_name="test-experiment",
        dataset_name="test-dataset",
        task_name="mock_task",
        raw_tags=["env=test", "version=1.0"],
        evaluator_names=["mock_evaluator"],
    )


def test_run_command_with_multiple_evaluators(cli_runner, mock_run_command):
    """Test the run command with multiple evaluators."""
    result = cli_runner.invoke(
        cli,
        [
            "run",
            "--name",
            "test-experiment",
            "--dataset",
            "test-dataset",
            "--task",
            "mock_task",
            "--evaluator",
            "mock_evaluator",
            "--evaluator",
            "mock_evaluator",
        ],
    )
    assert result.exit_code == 0
    mock_run_command.execute.assert_called_once_with(
        experiment_name="test-experiment",
        dataset_name="test-dataset",
        task_name="mock_task",
        raw_tags=None,
        evaluator_names=["mock_evaluator", "mock_evaluator"],
    )


def test_run_command_missing_required_args(cli_runner):
    """Test the run command fails appropriately when missing required arguments."""
    result = cli_runner.invoke(cli, ["run"])
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_run_command_invalid_task(cli_runner):
    """Test the run command fails appropriately with an invalid task."""
    result = cli_runner.invoke(
        cli,
        [
            "run",
            "--name",
            "test-experiment",
            "--dataset",
            "test-dataset",
            "--task",
            "invalid_task",
            "--evaluator",
            "mock_evaluator",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--task'" in result.output


def test_run_command_invalid_evaluator(cli_runner):
    """Test the run command fails appropriately with an invalid evaluator."""
    result = cli_runner.invoke(
        cli,
        [
            "run",
            "--name",
            "test-experiment",
            "--dataset",
            "test-dataset",
            "--task",
            "mock_task",
            "--evaluator",
            "invalid_evaluator",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--evaluator'" in result.output


@patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
def test_cli_respects_log_level(cli_runner):
    """Test that the CLI respects the LOG_LEVEL environment variable."""
    with patch("logging.basicConfig") as mock_basic_config:
        result = cli_runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        mock_basic_config.assert_called_once()
        assert mock_basic_config.call_args[1]["level"] == 10  # DEBUG level


def test_cli_handles_dotenv_error(cli_runner):
    """Test that the CLI handles dotenv loading errors gracefully."""
    with patch("arize_experiment.cli.cli.load_dotenv") as mock_load_dotenv:
        mock_load_dotenv.side_effect = Exception("Failed to load .env file")
        result = cli_runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 1
        assert "An unexpected error occurred: Failed to load .env file" in result.output


def test_create_dataset_command_success(cli_runner, mock_create_dataset_command):
    """Test the create_dataset command with valid arguments."""
    result = cli_runner.invoke(
        cli,
        [
            "create-dataset",
            "--name",
            "test-dataset",
            "--path-to-csv",
            "test.csv",
        ],
    )
    assert result.exit_code == 0
    mock_create_dataset_command.execute.assert_called_once_with(
        dataset_name="test-dataset",
        path_to_csv="test.csv",
    )


def test_create_dataset_command_missing_required_args(cli_runner):
    """Test that create_dataset command fails when missing required args."""
    result = cli_runner.invoke(cli, ["create-dataset"])
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_create_dataset_command_file_not_found(cli_runner, mock_create_dataset_command):
    """Test that create_dataset command fails when CSV file is not found."""
    mock_create_dataset_command.execute.side_effect = FileNotFoundError(
        "test.csv not found"
    )
    result = cli_runner.invoke(
        cli,
        [
            "create-dataset",
            "--name",
            "test-dataset",
            "--path-to-csv",
            "test.csv",
        ],
    )
    assert result.exit_code != 0
    assert "test.csv not found" in result.output

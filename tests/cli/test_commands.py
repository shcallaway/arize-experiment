"""
Tests for CLI commands in arize-experiment.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from arize_experiment.cli.commands import cli


@pytest.fixture
def cli_runner():
    """Fixture that provides a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_handler():
    """Fixture that provides a mocked Handler instance."""
    with patch("arize_experiment.cli.commands.Handler") as mock:
        handler_instance = MagicMock()
        mock.return_value = handler_instance
        yield handler_instance


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
    with (
        patch("arize_experiment.cli.commands.TaskRegistry.list") as mock_task_list,
        patch(
            "arize_experiment.cli.commands.EvaluatorRegistry.list"
        ) as mock_evaluator_list,
    ):
        mock_task_list.return_value = ["sentiment_classification"]
        mock_evaluator_list.return_value = ["sentiment_classification_accuracy"]
        yield


def test_cli_help(cli_runner):
    """Test that the CLI help command works."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "arize-experiment: A tool for running experiments on Arize" in result.output


def test_run_command_minimal(cli_runner, mock_handler):
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
            "sentiment_classification",
            "--evaluator",
            "sentiment_classification_accuracy",
        ],
    )
    if result.exit_code != 0:
        print(f"\nCommand output:\n{result.output}")
    assert result.exit_code == 0
    mock_handler.run.assert_called_once_with(
        experiment_name="test-experiment",
        dataset_name="test-dataset",
        task_name="sentiment_classification",
        raw_tags=None,
        evaluator_names=["sentiment_classification_accuracy"],
    )


def test_run_command_with_tags(cli_runner, mock_handler):
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
            "sentiment_classification",
            "--evaluator",
            "sentiment_classification_accuracy",
            "--tag",
            "env=test",
            "--tag",
            "version=1.0",
        ],
    )
    assert result.exit_code == 0
    mock_handler.run.assert_called_once_with(
        experiment_name="test-experiment",
        dataset_name="test-dataset",
        task_name="sentiment_classification",
        raw_tags=["env=test", "version=1.0"],
        evaluator_names=["sentiment_classification_accuracy"],
    )


def test_run_command_with_multiple_evaluators(cli_runner, mock_handler):
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
            "sentiment_classification",
            "--evaluator",
            "sentiment_classification_accuracy",
            "--evaluator",
            "sentiment_classification_accuracy",
        ],
    )
    assert result.exit_code == 0
    mock_handler.run.assert_called_once_with(
        experiment_name="test-experiment",
        dataset_name="test-dataset",
        task_name="sentiment_classification",
        raw_tags=None,
        evaluator_names=[
            "sentiment_classification_accuracy",
            "sentiment_classification_accuracy",
        ],
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
            "sentiment_classification_accuracy",
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
            "sentiment_classification",
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
    with patch("arize_experiment.cli.commands.load_dotenv") as mock_load_dotenv:
        mock_load_dotenv.side_effect = Exception("Failed to load .env file")
        result = cli_runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 1
        assert "An unexpected error occurred: Failed to load .env file" in result.output

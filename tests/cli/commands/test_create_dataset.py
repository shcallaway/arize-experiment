"""Tests for the create dataset command."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from arize_experiment.cli.commands.create_dataset import CreateDatasetCommand
from arize_experiment.core.errors import ConfigurationError, HandlerError


@pytest.fixture
def mock_arize_client():
    """Create a mock Arize client."""
    client = MagicMock()
    client.get_dataset.return_value = None  # Dataset doesn't exist by default
    client.create_dataset.return_value = "test_dataset_id"
    return client


@pytest.fixture
def command(mock_arize_client):
    """Create a CreateDatasetCommand instance with a mock Arize client."""
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
            command = CreateDatasetCommand()
            return command


def test_create_dataset_success(command, tmp_path):
    """Test successful dataset creation."""
    # Create a temporary CSV file
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame(
        {"column1": ["value1", "value2"], "column2": ["value3", "value4"]}
    )
    df.to_csv(csv_path, index=False, quoting=1, escapechar="\\")

    # Call execute
    command.execute("test_dataset", str(csv_path))

    # Verify the Arize client was called correctly
    command._arize_client.create_dataset.assert_called_once()
    call_args = command._arize_client.create_dataset.call_args[1]
    assert call_args["dataset_name"] == "test_dataset"
    assert isinstance(call_args["data"], pd.DataFrame)
    assert len(call_args["data"]) == 2


def test_create_dataset_already_exists(command, tmp_path):
    """Test error when dataset already exists."""
    # Mock that the dataset exists
    command._arize_client.get_dataset.return_value = {"id": "existing_id"}

    # Create a temporary CSV file
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"column1": ["value1"]})
    df.to_csv(csv_path, index=False)

    # Verify that attempting to create raises an error
    with pytest.raises(ConfigurationError) as exc_info:
        command.execute("test_dataset", str(csv_path))
    assert "already exists" in str(exc_info.value)


def test_create_dataset_empty_csv(command, tmp_path):
    """Test error when CSV file is empty."""
    # Create an empty CSV file
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w") as f:
        f.write("")

    # Verify that attempting to create raises an error
    with pytest.raises(HandlerError) as exc_info:
        command.execute("test_dataset", str(csv_path))
    assert "CSV file is empty" in str(exc_info.value)


def test_create_dataset_invalid_csv(command, tmp_path):
    """Test error when CSV file is invalid."""
    # Create an invalid CSV file
    csv_path = tmp_path / "invalid.csv"
    with open(csv_path, "w") as f:
        f.write('a,b\n1,"unclosed quote\n2,3')

    # Verify that attempting to create raises an error
    with pytest.raises(HandlerError) as exc_info:
        command.execute("test_dataset", str(csv_path))
    assert "Failed to parse CSV file" in str(exc_info.value)


def test_create_dataset_arize_error(command, tmp_path):
    """Test error when Arize client fails."""
    # Mock Arize client to raise an error
    command._arize_client.create_dataset.side_effect = Exception("Arize API error")

    # Create a valid CSV file
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"column1": ["value1"]})
    df.to_csv(csv_path, index=False)

    # Verify that attempting to create raises an error
    with pytest.raises(HandlerError) as exc_info:
        command.execute("test_dataset", str(csv_path))
    assert "Arize API error" in str(exc_info.value)


def test_create_dataset_no_dataset_id(command, tmp_path):
    """Test error when no dataset ID is returned."""
    # Mock Arize client to return None
    command._arize_client.create_dataset.return_value = None

    # Create a valid CSV file
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"column1": ["value1"]})
    df.to_csv(csv_path, index=False)

    # Verify that attempting to create raises an error
    with pytest.raises(HandlerError) as exc_info:
        command.execute("test_dataset", str(csv_path))
    assert "No dataset ID returned" in str(exc_info.value)

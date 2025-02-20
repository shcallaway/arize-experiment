"""Tests for the ArizeClient class."""

from unittest.mock import Mock, patch

import pytest

from arize_experiment.core.arize import ArizeClient, ArizeClientConfiguration
from arize_experiment.core.errors import ArizeClientError


@pytest.fixture
def mock_arize_datasets_client():
    """Mock the ArizeDatasetsClient."""
    with patch("arize_experiment.core.arize.ArizeDatasetsClient") as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client_config():
    """Create a test client configuration."""
    return ArizeClientConfiguration(
        api_key="test-api-key",
        developer_key="test-developer-key",
        space_id="test-space-id",
    )


@pytest.fixture
def arize_client(mock_arize_datasets_client, client_config):
    """Create a test ArizeClient instance."""
    return ArizeClient(config=client_config)


def test_client_initialization_success(mock_arize_datasets_client, client_config):
    """Test successful client initialization."""
    with patch("arize_experiment.core.arize.ArizeDatasetsClient") as mock_client_class:
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance

        client = ArizeClient(config=client_config)

        mock_client_class.assert_called_once_with(
            api_key=client_config.api_key,
            developer_key=client_config.developer_key,
        )
        assert client._space_id == client_config.space_id


def test_client_initialization_failure(client_config):
    """Test client initialization failure."""
    with patch("arize_experiment.core.arize.ArizeDatasetsClient") as mock_client:
        mock_client.side_effect = Exception("Connection failed")

        with pytest.raises(ArizeClientError) as exc_info:
            ArizeClient(config=client_config)

        assert "Failed to create Arize datasets client" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)


def test_get_dataset_success(arize_client, mock_arize_datasets_client):
    """Test successful dataset retrieval."""
    expected_dataset = {"name": "test-dataset"}
    mock_arize_datasets_client.get_dataset.return_value = expected_dataset

    result = arize_client.get_dataset("test-dataset")

    assert result == expected_dataset
    mock_arize_datasets_client.get_dataset.assert_called_once_with(
        space_id="test-space-id", dataset_name="test-dataset"
    )


def test_get_dataset_not_found(arize_client, mock_arize_datasets_client):
    """Test dataset not found scenario."""
    mock_arize_datasets_client.get_dataset.side_effect = Exception("Dataset not found")

    result = arize_client.get_dataset("non-existent-dataset")

    assert result is None


def test_get_dataset_error(arize_client, mock_arize_datasets_client):
    """Test dataset retrieval error."""
    mock_arize_datasets_client.get_dataset.side_effect = Exception("Unknown error")

    result = arize_client.get_dataset("test-dataset")

    assert result is None


def test_run_experiment_success(arize_client, mock_arize_datasets_client):
    """Test successful experiment execution."""
    expected_results = {"status": "success"}
    mock_arize_datasets_client.run_experiment.return_value = expected_results

    task_fn = Mock()
    evaluator_fn = Mock()

    result = arize_client.run_experiment(
        experiment_name="test-experiment",
        dataset_name="test-dataset",
        task=task_fn,
        evaluators=[evaluator_fn],
    )

    assert result == expected_results
    mock_arize_datasets_client.run_experiment.assert_called_once_with(
        space_id="test-space-id",
        dataset_name="test-dataset",
        task=task_fn,
        evaluators=[evaluator_fn],
        experiment_name="test-experiment",
    )


def test_run_experiment_failure(arize_client, mock_arize_datasets_client):
    """Test experiment execution failure."""
    mock_arize_datasets_client.run_experiment.side_effect = Exception(
        "Experiment failed"
    )

    with pytest.raises(ArizeClientError) as exc_info:
        arize_client.run_experiment(
            experiment_name="test-experiment", dataset_name="test-dataset", task=Mock()
        )

    assert "Failed to run experiment" in str(exc_info.value)
    assert "Experiment failed" in str(exc_info.value)


def test_get_experiment_success(arize_client, mock_arize_datasets_client):
    """Test successful experiment retrieval."""
    expected_experiment = {"name": "test-experiment"}
    mock_arize_datasets_client.get_experiment.return_value = expected_experiment

    result = arize_client.get_experiment(
        experiment_name="test-experiment", dataset_name="test-dataset"
    )

    assert result == expected_experiment
    mock_arize_datasets_client.get_experiment.assert_called_once_with(
        space_id="test-space-id",
        experiment_name="test-experiment",
        dataset_name="test-dataset",
    )


def test_get_experiment_not_found(arize_client, mock_arize_datasets_client):
    """Test experiment not found scenario."""
    mock_arize_datasets_client.get_experiment.side_effect = Exception("Not found")

    result = arize_client.get_experiment(
        experiment_name="non-existent", dataset_name="test-dataset"
    )

    assert result is None

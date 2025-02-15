"""Tests for configuration management."""

import os
import pytest
from unittest.mock import patch
from arize_experiment.config import (
    ArizeConfig,
    ExperimentConfig,
    EnvironmentError,
    load_environment,
    get_arize_config,
    create_experiment_config,
)


@pytest.fixture(autouse=True)
def mock_dotenv(monkeypatch):
    """Mock python-dotenv to prevent loading real .env files."""

    def mock_load_dotenv(*args, **kwargs):
        """Mock implementation that does nothing."""
        return True

    monkeypatch.setattr("arize_experiment.config.load_dotenv", mock_load_dotenv)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Fixture to ensure a clean environment for each test."""
    # Store original values
    original_api_key = os.environ.get("ARIZE_API_KEY")
    original_space_id = os.environ.get("ARIZE_SPACE_ID")

    # Clear environment variables
    monkeypatch.delenv("ARIZE_API_KEY", raising=False)
    monkeypatch.delenv("ARIZE_SPACE_ID", raising=False)

    yield monkeypatch

    # Restore original values if they existed
    if original_api_key is not None:
        monkeypatch.setenv("ARIZE_API_KEY", original_api_key)
    if original_space_id is not None:
        monkeypatch.setenv("ARIZE_SPACE_ID", original_space_id)


@pytest.fixture
def mock_env(clean_environment):
    """Fixture to set up test environment variables."""
    clean_environment.setenv("ARIZE_API_KEY", "test_api_key")
    clean_environment.setenv("ARIZE_SPACE_ID", "test_space_id")


def test_arize_config_creation():
    """Test ArizeConfig dataclass creation."""
    config = ArizeConfig(api_key="test_key", space_id="test_space", developer_key="test_dev_key")
    assert config.api_key == "test_key"
    assert config.space_id == "test_space"


def test_experiment_config_creation():
    """Test ExperimentConfig dataclass creation."""
    config = ExperimentConfig(
        name="test_exp",
        dataset="test_data",
        description="Test description",
        tags={"env": "test"},
    )
    assert config.name == "test_exp"
    assert config.dataset == "test_data"
    assert config.description == "Test description"
    assert config.tags == {"env": "test"}


def test_experiment_config_to_dict_minimal():
    """Test ExperimentConfig to_dict with minimal parameters."""
    config = ExperimentConfig(name="test_exp", dataset="test_data")
    result = config.to_dict()
    assert result == {
        "name": "test_exp",
        "dataset": "test_data",
    }


def test_experiment_config_to_dict_full():
    """Test ExperimentConfig to_dict with all parameters."""
    config = ExperimentConfig(
        name="test_exp",
        dataset="test_data",
        description="Test description",
        tags={"env": "test"},
    )
    result = config.to_dict()
    assert result == {
        "name": "test_exp",
        "dataset": "test_data",
        "description": "Test description",
        "tags": {"env": "test"},
    }


def test_get_arize_config_success(mock_env):
    """Test successful Arize configuration retrieval."""
    config = get_arize_config()
    assert isinstance(config, ArizeConfig)
    assert config.api_key == "test_api_key"
    assert config.space_id == "test_space_id"


def test_get_arize_config_missing_api_key(clean_environment):
    """Test error handling when API key is missing."""
    # Set only space_id
    clean_environment.setenv("ARIZE_SPACE_ID", "test_space_id")
    with pytest.raises(EnvironmentError) as exc_info:
        get_arize_config()
    assert "ARIZE_API_KEY environment variable is not set" in str(exc_info.value)


def test_get_arize_config_missing_space_id(clean_environment):
    """Test error handling when space ID is missing."""
    # Set only api_key
    clean_environment.setenv("ARIZE_API_KEY", "test_api_key")
    with pytest.raises(EnvironmentError) as exc_info:
        get_arize_config()
    assert "ARIZE_SPACE_ID environment variable is not set" in str(exc_info.value)


def test_create_experiment_config_minimal():
    """Test experiment config creation with minimal parameters."""
    config = create_experiment_config(name="test_exp", dataset="test_data")
    assert isinstance(config, ExperimentConfig)
    assert config.name == "test_exp"
    assert config.dataset == "test_data"
    assert config.description is None
    assert config.tags is None


def test_create_experiment_config_full():
    """Test experiment config creation with all parameters."""
    config = create_experiment_config(
        name="test_exp",
        dataset="test_data",
        description="Test description",
        tags={"env": "test"},
    )
    assert isinstance(config, ExperimentConfig)
    assert config.name == "test_exp"
    assert config.dataset == "test_data"
    assert config.description == "Test description"
    assert config.tags == {"env": "test"}


def test_load_environment(tmp_path, clean_environment):
    """Test environment loading from .env file."""
    with patch("arize_experiment.config.load_dotenv") as mock_load_dotenv:
        mock_load_dotenv.return_value = True
        load_environment()
        mock_load_dotenv.assert_called_once()

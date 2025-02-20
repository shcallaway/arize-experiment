"""Tests for schema validator integration and complex validation scenarios."""

from typing import Dict
from unittest.mock import MagicMock

from arize_experiment.core.schema import ColumnSchema, DatasetSchema, DataType
from arize_experiment.core.schema_validator import SchemaValidator
from arize_experiment.core.task import Task, TaskResult


class MockTask(Task):
    """Mock task for testing schema validation."""

    def __init__(self, schema: DatasetSchema) -> None:
        self._schema = schema

    @property
    def name(self) -> str:
        return "mock_task"

    @property
    def required_schema(self) -> DatasetSchema:
        return self._schema

    def execute(self, Input: Dict) -> TaskResult:
        return TaskResult(input=Input, output=None)


def test_schema_validator_integration():
    """Test SchemaValidator integration with Task and ArizeClient."""
    schema = DatasetSchema(
        columns={
            "input": ColumnSchema(name="input", types=[DataType.STRING], required=True)
        }
    )
    task = MockTask(schema)

    # Create mock Arize client
    mock_client = MagicMock()
    mock_dataset = MagicMock()
    mock_sample = MagicMock()

    # Valid data
    valid_data = {"input": "hello"}
    mock_sample.data = [valid_data]
    mock_dataset.get_sample.return_value = mock_sample
    mock_client.get_dataset.return_value = mock_dataset

    validator = SchemaValidator()
    errors = validator.validate("test_dataset", task, mock_client)
    assert not errors

    # Invalid data
    invalid_data = {"input": 123}  # Wrong type
    mock_sample.data = [invalid_data]
    mock_dataset.get_sample.return_value = mock_sample
    mock_client.get_dataset.return_value = mock_dataset

    errors = validator.validate("test_dataset", task, mock_client)
    assert len(errors) == 1
    assert errors[0].path == "input"
    assert "type" in errors[0].message.lower()

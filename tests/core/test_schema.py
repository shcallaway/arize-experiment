"""Tests for schema definition and basic validation functionality."""

from typing import Any, Dict

from arize_experiment.core.schema import ColumnSchema, DatasetSchema, DataType


def test_simple_schema_validation():
    """Test validation of a simple flat schema."""
    schema = DatasetSchema(
        columns={
            "text": ColumnSchema(name="text", types=[DataType.STRING], required=True),
            "score": ColumnSchema(
                name="score", types=[DataType.INTEGER, DataType.FLOAT], required=True
            ),
            "optional_field": ColumnSchema(
                name="optional_field", types=[DataType.STRING], required=False
            ),
        }
    )

    # Valid data
    valid_data = {
        "text": "hello",
        "score": 0.95,
    }
    assert not schema.validate_data(valid_data)

    # Missing required field
    missing_field_data: Dict[str, Any] = {"text": "hello"}
    errors = schema.validate_data(missing_field_data)
    assert len(errors) == 1
    assert errors[0].path == "score"
    assert "missing" in errors[0].message.lower()

    # Wrong type
    wrong_type_data: Dict[str, Any] = {
        "text": 123,
        "score": 0.95,
    }  # text should be string
    errors = schema.validate_data(wrong_type_data)
    assert len(errors) == 1
    assert errors[0].path == "text"
    assert "type" in errors[0].message.lower()


def test_nested_schema_validation():
    """Test validation of nested schema structures."""
    schema = DatasetSchema(
        columns={
            "messages": ColumnSchema(
                name="messages",
                types=[DataType.LIST],
                nested_schema={
                    "role": ColumnSchema(
                        name="role", types=[DataType.STRING], required=True
                    ),
                    "content": ColumnSchema(
                        name="content", types=[DataType.STRING], required=True
                    ),
                },
            )
        }
    )

    # Valid data
    valid_data = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    }
    assert not schema.validate_data(valid_data)

    # Invalid nested data
    invalid_data = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant"},  # Missing content
        ]
    }
    errors = schema.validate_data(invalid_data)
    assert len(errors) == 1
    assert "messages[1].content" in errors[0].path
    assert "missing" in errors[0].message.lower()


def test_union_types():
    """Test validation of union types."""
    schema = DatasetSchema(
        columns={
            "value": ColumnSchema(
                name="value",
                types=[DataType.INTEGER, DataType.FLOAT, DataType.NULL],
                required=True,
            )
        }
    )

    # Test all valid types
    assert not schema.validate_data({"value": 42})
    assert not schema.validate_data({"value": 3.14})
    assert not schema.validate_data({"value": None})

    # Test invalid type
    errors = schema.validate_data({"value": "string"})
    assert len(errors) == 1
    assert errors[0].path == "value"
    assert "type" in errors[0].message.lower()


def test_dict_validation():
    """Test validation of dictionary structures."""
    schema = DatasetSchema(
        columns={
            "metadata": ColumnSchema(
                name="metadata",
                types=[DataType.DICT],
                nested_schema={
                    "id": ColumnSchema(
                        name="id", types=[DataType.STRING], required=True
                    ),
                    "tags": ColumnSchema(
                        name="tags", types=[DataType.LIST], required=False
                    ),
                },
            )
        }
    )

    # Valid data
    valid_data = {"metadata": {"id": "123", "tags": ["test", "example"]}}
    assert not schema.validate_data(valid_data)

    # Missing optional field is ok
    valid_data = {"metadata": {"id": "123"}}
    assert not schema.validate_data(valid_data)

    # Invalid nested type
    invalid_data = {"metadata": {"id": 123, "tags": ["test"]}}  # Should be string
    errors = schema.validate_data(invalid_data)
    assert len(errors) == 1
    assert "metadata.id" in errors[0].path
    assert "type" in errors[0].message.lower()

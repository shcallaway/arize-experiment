"""
Schema definitions and validation for Arize datasets.

This module provides the core schema types and validation logic used to ensure
dataset compatibility with tasks. It supports:
1. Rich type definitions
2. Nested schema structures
3. Required/optional fields
4. Schema validation
5. Detailed error reporting

The schema system is designed to be:
1. Flexible - Supporting various data types and structures
2. Type-safe - Ensuring data consistency
3. Extensible - Easy to add new types
4. Self-documenting - Clear type definitions

Example:
    ```python
    from arize_experiment.core.schema import DatasetSchema, ColumnSchema, DataType

    # Define a schema for user data
    user_schema = DatasetSchema(
        columns={
            "name": ColumnSchema(
                name="name",
                types=[DataType.STRING],
                required=True,
                description="User's full name"
            ),
            "age": ColumnSchema(
                name="age",
                types=[DataType.INTEGER],
                required=False,
                description="User's age in years"
            )
        },
        description="Schema for user profile data"
    )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DataType(Enum):
    """Supported data types for schema validation.

    This enum defines the valid data types that can be used in dataset schemas.
    Each type corresponds to a Python type and is used for validating data
    before task execution.

    The type system supports:
    1. Basic types (string, integer, float, boolean)
    2. Collection types (list, dict)
    3. Special types (null)
    4. Type unions (via lists of types)

    Attributes:
        STRING: String data type (str)
        INTEGER: Integer data type (int)
        FLOAT: Floating point data type (float)
        BOOLEAN: Boolean data type (bool)
        LIST: List data type (list)
        DICT: Dictionary data type (dict)
        NULL: Null/None data type

    Example:
        ```python
        column = ColumnSchema(
            name="age",
            types=[DataType.INTEGER, DataType.NULL],  # Allows integers or None
            required=True
        )
        ```
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    NULL = "null"


@dataclass
class ValidationError:
    """Represents a schema validation error."""

    path: str
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


@dataclass
class ColumnSchema:
    """Schema definition for a single column or nested field.

    This class defines the schema for a single column in a dataset, including
    its name, allowed data types, and whether it's required. It also supports
    nested schemas for complex data types like dictionaries and lists.

    Features:
    1. Multiple allowed types per column
    2. Required/optional fields
    3. Nested schema support
    4. Field descriptions
    5. Type validation

    Attributes:
        name (str): The name of the column
        types (List[DataType]): List of allowed data types for this column
        required (bool): Whether this column is required (default: True)
        nested_schema (Optional[Dict[str, ColumnSchema]]): Schema for nested fields
        description (Optional[str]): Human-readable description of the column

    Example:
        ```python
        # Simple string column
        name_schema = ColumnSchema(
            name="name",
            types=[DataType.STRING],
            required=True,
            description="User's full name"
        )

        # Nested dictionary column with complex structure
        address_schema = ColumnSchema(
            name="address",
            types=[DataType.DICT],
            nested_schema={
                "street": ColumnSchema(
                    name="street",
                    types=[DataType.STRING],
                    description="Street address"
                ),
                "city": ColumnSchema(
                    name="city",
                    types=[DataType.STRING],
                    description="City name"
                ),
                "coordinates": ColumnSchema(
                    name="coordinates",
                    types=[DataType.DICT],
                    nested_schema={
                        "lat": ColumnSchema(
                            name="lat",
                            types=[DataType.FLOAT],
                            description="Latitude"
                        ),
                        "lng": ColumnSchema(
                            name="lng",
                            types=[DataType.FLOAT],
                            description="Longitude"
                        )
                    }
                )
            }
        )
        ```
    """

    name: str
    types: List[DataType]  # Supports union types
    required: bool = True
    nested_schema: Optional[Dict[str, ColumnSchema]] = field(default=None)
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate schema configuration after initialization.

        This method ensures that:
        1. Nested schemas are only used with appropriate types
        2. Type lists are valid
        3. Names follow conventions

        Raises:
            ValueError: If schema configuration is invalid
        """
        if (
            self.nested_schema is not None
            and DataType.DICT not in self.types
            and DataType.LIST not in self.types
        ):
            raise ValueError(
                f"Column {self.name} has nested schema but type is not DICT or LIST"
            )


@dataclass
class DatasetSchema:
    """Schema definition for an entire dataset.

    This class defines the complete schema for a dataset, including all its
    columns and their requirements. It provides methods for validating data
    against the schema and reporting any validation errors.

    Features:
    1. Column definitions
    2. Nested structure support
    3. Data validation
    4. Error reporting
    5. Schema documentation

    Attributes:
        columns (Dict[str, ColumnSchema]): Map of column names to their schemas
        description (Optional[str]): Human-readable description of the dataset

    Example:
        ```python
        # Define a schema for user profiles
        user_schema = DatasetSchema(
            columns={
                "id": ColumnSchema(
                    name="id",
                    types=[DataType.STRING],
                    required=True,
                    description="Unique user identifier"
                ),
                "profile": ColumnSchema(
                    name="profile",
                    types=[DataType.DICT],
                    nested_schema={
                        "name": ColumnSchema(
                            name="name",
                            types=[DataType.STRING]
                        ),
                        "age": ColumnSchema(
                            name="age",
                            types=[DataType.INTEGER],
                            required=False
                        )
                    }
                )
            },
            description="Schema for user profile data"
        )
        ```
    """

    columns: Dict[str, ColumnSchema]
    description: Optional[str] = None

    def validate_data(self, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate data against this schema.

        This method performs comprehensive validation including:
        1. Required field presence
        2. Type checking
        3. Nested structure validation
        4. Custom validation rules

        Args:
            data: Dictionary of data to validate

        Returns:
            List of validation errors, empty if validation passes

        Example:
            ```python
            errors = schema.validate_data({
                "id": "user123",
                "profile": {
                    "name": "John Doe",
                    "age": 30
                }
            })
            if errors:
                print("Validation failed:", errors)
            ```
        """
        errors: List[ValidationError] = []
        self._validate_data_recursive(data, "", errors)
        return errors

    def _validate_type(
        self, value: Any, allowed_types: List[DataType], path: str, column_name: str
    ) -> Optional[ValidationError]:
        """Validate the type of a value against allowed types.

        Args:
            value: The value to validate
            allowed_types: List of allowed DataTypes
            path: Current path in data structure
            column_name: Name of the column being validated

        Returns:
            ValidationError if validation fails, None otherwise
        """
        if value is None:
            if DataType.NULL not in allowed_types:
                return ValidationError(
                    path=path,
                    message=f"Field '{column_name}' cannot be null",
                    expected=str([t.value for t in allowed_types]),
                    actual="null",
                )
            return None

        type_matches = (
            (DataType.STRING in allowed_types and isinstance(value, str))
            or (DataType.INTEGER in allowed_types and isinstance(value, int))
            or (DataType.FLOAT in allowed_types and isinstance(value, (int, float)))
            or (DataType.BOOLEAN in allowed_types and isinstance(value, bool))
            or (DataType.LIST in allowed_types and isinstance(value, list))
            or (DataType.DICT in allowed_types and isinstance(value, dict))
        )

        if not type_matches:
            return ValidationError(
                path=path,
                message=f"Invalid type for field '{column_name}'",
                expected=str([t.value for t in allowed_types]),
                actual=type(value).__name__,
            )
        return None

    def _validate_nested_structure(
        self,
        value: Any,
        schema: Dict[str, ColumnSchema],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        """Validate a nested dictionary or list structure.

        Args:
            value: The value to validate
            schema: The nested schema to validate against
            path: Current path in data structure
            errors: List to collect validation errors
        """
        nested_schema = DatasetSchema(columns=schema)
        if isinstance(value, dict):
            nested_schema._validate_data_recursive(value, path, errors)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if not isinstance(item, dict):
                    errors.append(
                        ValidationError(
                            path=f"{path}[{i}]",
                            message="List item must be a dictionary",
                            expected="dict",
                            actual=type(item).__name__,
                        )
                    )
                    continue
                nested_schema._validate_data_recursive(item, f"{path}[{i}]", errors)

    def _validate_data_recursive(
        self, data: Dict[str, Any], path: str, errors: List[ValidationError]
    ) -> None:
        """Recursively validate nested data structures.

        Args:
            data: The data to validate
            path: Current path in the data structure
            errors: List to collect validation errors
        """
        for column_name, column_schema in self.columns.items():
            current_path = f"{path}.{column_name}" if path else column_name

            # Check required fields
            if column_name not in data:
                if column_schema.required:
                    errors.append(
                        ValidationError(
                            path=current_path,
                            message=f"Required field '{column_name}' is missing",
                        )
                    )
                continue

            value = data[column_name]

            # Validate type
            type_error = self._validate_type(
                value, column_schema.types, current_path, column_name
            )
            if type_error:
                errors.append(type_error)
                continue

            # Validate nested structures
            if column_schema.nested_schema and value is not None:
                self._validate_nested_structure(
                    value, column_schema.nested_schema, current_path, errors
                )

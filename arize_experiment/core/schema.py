from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, cast


class DataType(Enum):
    """Supported data types for schema validation."""

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
    """Schema definition for a single column or nested field."""

    name: str
    types: List[DataType]  # Supports union types
    required: bool = True
    nested_schema: Optional[Dict[str, ColumnSchema]] = field(default=None)
    description: Optional[str] = None

    def __post_init__(self) -> None:
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
    """Schema definition for a dataset."""

    columns: Dict[str, ColumnSchema]
    version: str = "1.0.0"
    description: Optional[str] = None

    def validate_data(self, data: Dict) -> List[ValidationError]:
        """Validate data against this schema.

        Args:
            data: Dictionary containing the data to validate

        Returns:
            List of ValidationError objects if validation fails, empty list if successful
        """
        errors: List[ValidationError] = []
        self._validate_dict(data, self.columns, "", errors)
        return errors

    def _validate_dict(
        self,
        data: Dict,
        schema: Dict[str, ColumnSchema],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        """Recursively validate a dictionary against a schema."""
        # Check for required fields
        for name, col_schema in schema.items():
            field_path = f"{path}.{name}" if path else name

            if name not in data:
                if col_schema.required:
                    errors.append(
                        ValidationError(
                            path=field_path,
                            message="Required field is missing",
                            expected=str([t.value for t in col_schema.types]),
                            actual="missing",
                        )
                    )
                continue

            value = data[name]
            self._validate_value(value, col_schema, field_path, errors)

    def _validate_value(
        self, value: Any, schema: ColumnSchema, path: str, errors: List[ValidationError]
    ) -> None:
        """Validate a single value against its schema."""
        # Handle null values
        if value is None:
            if DataType.NULL not in schema.types:
                errors.append(
                    ValidationError(
                        path=path,
                        message="Null value not allowed",
                        expected=str([t.value for t in schema.types]),
                        actual="null",
                    )
                )
            return

        # Get Python type mapping
        type_mapping = {
            DataType.STRING: str,
            DataType.INTEGER: int,
            DataType.FLOAT: (int, float),  # Allow integers for float fields
            DataType.BOOLEAN: bool,
            DataType.LIST: list,
            DataType.DICT: dict,
        }

        # Check type
        valid_types: List[type] = []
        for dtype in schema.types:
            if dtype == DataType.NULL:
                continue
            mapped_type = type_mapping[dtype]
            if isinstance(mapped_type, tuple):
                valid_types.extend(list(mapped_type))
            else:
                valid_types.append(cast(type, mapped_type))

        if not isinstance(value, tuple(valid_types)):
            errors.append(
                ValidationError(
                    path=path,
                    message="Invalid type",
                    expected=str([t.value for t in schema.types]),
                    actual=type(value).__name__,
                )
            )
            return

        # Validate nested structures
        if schema.nested_schema:
            if isinstance(value, dict):
                self._validate_dict(value, schema.nested_schema, path, errors)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if not isinstance(item, dict):
                        errors.append(
                            ValidationError(
                                path=f"{path}[{i}]",
                                message="List item must be a dictionary for nested schema validation",
                                expected="dict",
                                actual=type(item).__name__,
                            )
                        )
                        continue
                    self._validate_dict(
                        item, schema.nested_schema, f"{path}[{i}]", errors
                    )

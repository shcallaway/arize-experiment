"""
Tests for the configurable evaluator.
"""

from typing import Any

import pytest
from arize.experimental.datasets.experiments.types import EvaluationResult

from arize_experiment.core.configurable_evaluator import ConfigurableEvaluator
from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry


@EvaluatorRegistry.register("configurable_test")
class ConfigurableTestEvaluator(BaseEvaluator):
    """Test evaluator for configuration testing."""

    def __init__(self, score: float = 1.0, label: str = "test"):
        super().__init__()
        self.score = score
        self.label = label

    @property
    def name(self) -> str:
        return "configurable_test"

    def evaluate(self, output: Any) -> EvaluationResult:
        return EvaluationResult(score=self.score, label=self.label)

    def __call__(self, output: Any) -> EvaluationResult:
        return self.evaluate(output)


def test_configurable_evaluator_creation():
    """Test creating an evaluator from a valid configuration."""
    config = {"type": "configurable_test", "score": 0.8, "label": "custom_label"}

    evaluator = ConfigurableEvaluator.from_config(config)
    assert isinstance(evaluator, ConfigurableTestEvaluator)
    assert evaluator.score == 0.8
    assert evaluator.label == "custom_label"


def test_configurable_evaluator_defaults():
    """Test that default values are used when not specified in config."""
    config = {"type": "configurable_test"}

    evaluator = ConfigurableEvaluator.from_config(config)
    assert isinstance(evaluator, ConfigurableTestEvaluator)
    assert evaluator.score == 1.0
    assert evaluator.label == "test"


def test_invalid_evaluator_type():
    """Test that using an invalid evaluator type raises an error."""
    config = {"type": "nonexistent_evaluator"}

    with pytest.raises(ValueError):
        ConfigurableEvaluator.from_config(config)


def test_missing_type():
    """Test that missing type in config raises an error."""
    config = {"score": 0.8, "label": "test"}

    with pytest.raises(ValueError):
        ConfigurableEvaluator.from_config(config)


def test_invalid_config_type():
    """Test that non-dict configs raise an error."""
    invalid_configs: list[Any] = [
        None,
        123,
        ["list", "not", "dict"],
    ]

    for config in invalid_configs:
        with pytest.raises(ValueError):
            ConfigurableEvaluator.from_config(config)


def test_invalid_parameters():
    """Test that invalid parameters raise an error."""
    config = {"type": "configurable_test", "invalid_param": "value"}

    with pytest.raises(ValueError):
        ConfigurableEvaluator.from_config(config)

"""
Tests for the experiment module.
"""

import pytest
from typing import Any, Dict
from unittest.mock import Mock

from arize.experimental.datasets.experiments.types import EvaluationResult

from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize_experiment.core.experiment import Experiment
from arize_experiment.core.task import Task, TaskResult


class MockTask(Task):
    """Mock task for testing."""

    @property
    def name(self) -> str:
        return "mock_task"

    def execute(self, Input: Dict[str, Any]) -> TaskResult:
        return TaskResult(input=Input, output="mock_output")


@EvaluatorRegistry.register("mock_evaluator")
class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing."""

    def __init__(self, score: float = 1.0, label: str = "mock_label"):
        self.score = score
        self.label = label

    @property
    def name(self) -> str:
        return "mock_evaluator"

    def evaluate(self, output: Any) -> EvaluationResult:
        return EvaluationResult(score=self.score, label=self.label)

    def __call__(self, output: Any) -> EvaluationResult:
        return self.evaluate(output)


def test_experiment_to_dict() -> None:
    """Test that Experiment.to_dict() returns the expected dictionary format."""
    # Arrange
    evaluator_configs = [
        {"type": "mock_evaluator", "score": 1.0, "label": "mock_1"},
        {"type": "mock_evaluator", "score": 0.5, "label": "mock_2"},
    ]
    
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluator_configs=evaluator_configs,
        description="Test description",
        tags={"env": "test", "version": "1.0"},
    )

    # Act
    result = experiment.to_dict()

    # Assert
    assert result == {
        "name": "test_experiment",
        "dataset": "test_dataset",
        "task": "mock_task",
        "evaluator_configs": evaluator_configs,
        "description": "Test description",
        "tags": {"env": "test", "version": "1.0"},
    }


def test_experiment_to_dict_minimal() -> None:
    """Test that Experiment.to_dict() works with minimal required fields."""
    # Arrange
    evaluator_configs = [
        {"type": "mock_evaluator", "score": 1.0, "label": "mock_label"},
    ]
    
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluator_configs=evaluator_configs,
    )

    # Act
    result = experiment.to_dict()

    # Assert
    assert result == {
        "name": "test_experiment",
        "dataset": "test_dataset",
        "task": "mock_task",
        "evaluator_configs": evaluator_configs,
        "description": None,
        "tags": {},
    }


def test_experiment_init() -> None:
    """Test experiment initialization."""
    evaluator_configs = [
        {"type": "mock_evaluator", "score": 1.0, "label": "mock_label"},
    ]
    
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluator_configs=evaluator_configs,
    )
    
    assert experiment.task is not None
    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], MockEvaluator)


def test_experiment_invalid_evaluator_config() -> None:
    """Test experiment initialization with invalid evaluator config."""
    invalid_configs = [
        {"type": "nonexistent_evaluator"},
    ]
    
    with pytest.raises(ValueError):
        Experiment(
            name="test_experiment",
            dataset="test_dataset",
            task=MockTask(),
            evaluator_configs=invalid_configs,
        )


def test_experiment_evaluator_execution() -> None:
    """Test that configured evaluators work correctly."""
    evaluator_configs = [
        {"type": "mock_evaluator", "score": 0.8, "label": "test_label"},
    ]
    
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluator_configs=evaluator_configs,
    )
    
    # Get the configured evaluator
    evaluator = experiment.evaluators[0]
    assert isinstance(evaluator, MockEvaluator)
    
    # Test evaluation
    result = evaluator.evaluate("test_output")
    assert result.score == 0.8
    assert result.label == "test_label"

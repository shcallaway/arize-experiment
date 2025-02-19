"""
Tests for the experiment module.
"""

import pytest
from typing import Any, Dict

from arize.experimental.datasets.experiments.types import EvaluationResult
from arize_experiment.core.experiment import Experiment
from arize_experiment.core.task import Task, TaskResult
from arize_experiment.core.evaluator import BaseEvaluator


class MockTask(Task):
    """Mock task for testing."""
    
    @property
    def name(self) -> str:
        return "mock_task"
    
    def execute(self, Input: Dict[str, Any]) -> TaskResult:
        return TaskResult(input=Input, output="mock_output")


class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing."""
    
    @property
    def name(self) -> str:
        return "mock_evaluator"
    
    def evaluate(self, output: Any) -> EvaluationResult:
        return EvaluationResult(score=1.0, label="mock_label")
    
    def __call__(self, output: Any) -> EvaluationResult:
        return self.evaluate(output)


def test_experiment_to_dict():
    """Test that Experiment.to_dict() returns the expected dictionary format."""
    # Arrange
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluators=[MockEvaluator(), MockEvaluator()],
        description="Test description",
        tags={"env": "test", "version": "1.0"}
    )
    
    # Act
    result = experiment.to_dict()
    
    # Assert
    assert result == {
        "name": "test_experiment",
        "dataset": "test_dataset",
        "task": "mock_task",
        "evaluators": ["mock_evaluator", "mock_evaluator"],
        "description": "Test description",
        "tags": {"env": "test", "version": "1.0"}
    }


def test_experiment_to_dict_minimal():
    """Test that Experiment.to_dict() works with minimal required fields."""
    # Arrange
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluators=[MockEvaluator()],
    )
    
    # Act
    result = experiment.to_dict()
    
    # Assert
    assert result == {
        "name": "test_experiment",
        "dataset": "test_dataset",
        "task": "mock_task",
        "evaluators": ["mock_evaluator"],
        "description": None,
        "tags": {}
    } 

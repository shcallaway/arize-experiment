"""
Tests for the experiment module.
"""

from typing import Any, Dict, Generator, Type

import pytest
from arize.experimental.datasets.experiments.types import EvaluationResult

from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize_experiment.core.experiment import Experiment
from arize_experiment.core.task import Task, TaskResult


@pytest.fixture(autouse=True)
def mock_evaluator_class() -> Generator[Type[BaseEvaluator], None, None]:
    """Fixture providing the mock evaluator class.

    This fixture also handles clearing and restoring the evaluator registry.
    """
    # Store original registry state
    original_evaluators = EvaluatorRegistry._evaluators.copy()

    # Clear registry
    EvaluatorRegistry._evaluators.clear()

    # Create and register mock evaluator
    @EvaluatorRegistry.register("mock_evaluator")
    class MockEvaluator(BaseEvaluator):
        """Mock evaluator for testing."""

        def __init__(self, score: float = 1.0, label: str = "mock_label"):
            super().__init__()
            self.score = score
            self.label = label

        @property
        def name(self) -> str:
            return "mock_evaluator"

        def evaluate(self, output: Any) -> EvaluationResult:
            return EvaluationResult(score=self.score, label=self.label)

        def __call__(self, output: Any) -> EvaluationResult:
            return self.evaluate(output)

    yield MockEvaluator

    # Restore original registry state
    EvaluatorRegistry._evaluators.clear()
    EvaluatorRegistry._evaluators.update(original_evaluators)


class MockTask(Task):
    """Mock task for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the mock task."""
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return "mock_task"

    def execute(self, Input: Dict[str, Any]) -> TaskResult:
        return TaskResult(input=Input, output="mock_output")


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


def test_experiment_init(mock_evaluator_class: Type[BaseEvaluator]) -> None:
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
    assert isinstance(experiment.evaluators[0], mock_evaluator_class)


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


def test_experiment_evaluator_execution(
    mock_evaluator_class: Type[BaseEvaluator],
) -> None:
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
    assert isinstance(evaluator, mock_evaluator_class)

    # Test evaluation
    result = evaluator.evaluate("test_output")
    assert result.score == 0.8
    assert result.label == "test_label"


def test_experiment_multiple_evaluators(
    mock_evaluator_class: Type[BaseEvaluator],
) -> None:
    """Test experiment with multiple evaluators."""
    evaluator_configs = [
        {"type": "mock_evaluator", "score": 0.8, "label": "evaluator_1"},
        {"type": "mock_evaluator", "score": 0.6, "label": "evaluator_2"},
        {"type": "mock_evaluator", "score": 0.4, "label": "evaluator_3"},
    ]

    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluator_configs=evaluator_configs,
    )

    # Verify all evaluators are initialized
    assert len(experiment.evaluators) == 3

    # Verify each evaluator's configuration
    for i, evaluator in enumerate(experiment.evaluators):
        assert isinstance(evaluator, mock_evaluator_class)
        result = evaluator.evaluate("test_output")
        assert result.score == evaluator_configs[i]["score"]
        assert result.label == evaluator_configs[i]["label"]

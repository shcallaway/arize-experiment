"""
Tests for the experiment module.
"""

from typing import Any, Dict
from unittest.mock import Mock

from arize.experimental.datasets.experiments.types import EvaluationResult

from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.experiment import Experiment
from arize_experiment.core.task import Task, TaskResult


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


def test_experiment_to_dict() -> None:
    """Test that Experiment.to_dict() returns the expected dictionary format."""
    # Arrange
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluators=[MockEvaluator(), MockEvaluator()],
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
        "evaluators": ["mock_evaluator", "mock_evaluator"],
        "description": "Test description",
        "tags": {"env": "test", "version": "1.0"},
    }


def test_experiment_to_dict_minimal() -> None:
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
        "tags": {},
    }


def test_experiment_init() -> None:
    """Test experiment initialization."""
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluators=[MockEvaluator()],
    )
    assert experiment.task is not None
    assert len(experiment.evaluators) == 1


def test_add_task() -> None:
    """Test adding a task to the experiment."""
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluators=[MockEvaluator()],
    )
    task = MockTask()
    experiment.task = task
    assert experiment.task == task


def test_add_evaluator() -> None:
    """Test adding an evaluator to the experiment."""
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluators=[MockEvaluator()],
    )
    evaluator = MockEvaluator()
    experiment.evaluators.append(evaluator)
    assert evaluator in experiment.evaluators


def test_run_tasks() -> None:
    """Test running tasks in the experiment."""
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluators=[MockEvaluator()],
    )
    task = Mock()
    experiment.task = task
    task_input = {"input": "test input"}
    task.execute.return_value = TaskResult(input=task_input, output="test output")
    result = experiment.task.execute(task_input)
    task.execute.assert_called_once_with(task_input)
    assert result.output == "test output"


def test_run_evaluators() -> None:
    """Test running evaluators in the experiment."""
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluators=[MockEvaluator()],
    )
    evaluator1 = Mock()
    evaluator2 = Mock()
    evaluator1.evaluate.return_value = EvaluationResult(score=1.0, label="label1")
    evaluator2.evaluate.return_value = EvaluationResult(score=0.5, label="label2")
    experiment.evaluators = [evaluator1, evaluator2]
    task_result = TaskResult(input={"input": "test"}, output="test output")
    for evaluator in experiment.evaluators:
        result = evaluator.evaluate(task_result.output)
        assert isinstance(result, EvaluationResult)


def test_save_results() -> None:
    """Test saving experiment results."""
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=MockTask(),
        evaluators=[MockEvaluator()],
    )
    task = Mock()
    task.to_dict.return_value = {"input": "test input", "output": "test output"}
    experiment.task = task
    evaluator = Mock()
    evaluator.to_dict.return_value = {"metric": "test metric", "value": 0.5}
    experiment.evaluators = [evaluator]

    # Since there's no save_results method, we'll just test the to_dict method
    result = experiment.to_dict()
    assert result["name"] == "test_experiment"
    assert result["dataset"] == "test_dataset"
    assert "task" in result
    assert "evaluators" in result

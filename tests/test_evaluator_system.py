"""
Tests for the new evaluator system.
"""

import pytest

from arize_experiment.core.configurable_evaluator import ConfigurableEvaluator
from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize_experiment.core.experiment import Experiment
from arize_experiment.core.task import Task, TaskResult
from arize_experiment.evaluators.sentiment_classification_accuracy import (
    SentimentClassificationAccuracyEvaluator,
)


class MockTask(Task):
    """Mock task for testing."""
    
    @property
    def name(self) -> str:
        return "mock_task"
    
    def execute(self, input_data: str) -> TaskResult:
        return TaskResult(
            input={"input": input_data},  # Match the expected input format
            output="positive",
            success=True,
            error=None
        )


def test_evaluator_registry():
    """Test the evaluator registry functionality."""
    # Test registration
    assert "sentiment_classification_accuracy" in EvaluatorRegistry._evaluators
    
    # Test retrieval
    evaluator_class = EvaluatorRegistry.get("sentiment_classification_accuracy")
    assert evaluator_class == SentimentClassificationAccuracyEvaluator
    
    # Test invalid retrieval
    with pytest.raises(ValueError):
        EvaluatorRegistry.get("nonexistent_evaluator")


def test_configurable_evaluator():
    """Test the configurable evaluator functionality."""
    # Test valid configuration
    config = {
        "type": "sentiment_classification_accuracy",
        "model": "gpt-4",
        "temperature": 0.5,
        "api_key": "test-key"
    }
    
    evaluator = ConfigurableEvaluator.from_config(config)
    assert isinstance(evaluator, SentimentClassificationAccuracyEvaluator)
    assert evaluator._model == "gpt-4"
    assert evaluator._temperature == 0.5
    assert evaluator._client.api_key == "test-key"
    
    # Test invalid configuration
    with pytest.raises(ValueError):
        ConfigurableEvaluator.from_config({"type": "nonexistent"})
    
    with pytest.raises(ValueError):
        ConfigurableEvaluator.from_config({})


def test_experiment_with_evaluators():
    """Test experiment integration with the new evaluator system."""
    task = MockTask()
    evaluator_configs = [
        {
            "type": "sentiment_classification_accuracy",
            "model": "gpt-4",
            "temperature": 0.0,
            "api_key": "test-key"
        }
    ]
    
    # Create experiment
    experiment = Experiment(
        name="test_experiment",
        dataset="test_dataset",
        task=task,
        evaluator_configs=evaluator_configs
    )
    
    # Verify evaluators are initialized
    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], SentimentClassificationAccuracyEvaluator)
    
    # Test serialization
    exp_dict = experiment.to_dict()
    assert exp_dict["evaluator_configs"] == evaluator_configs


def test_experiment_invalid_config():
    """Test experiment creation with invalid evaluator config."""
    task = MockTask()
    invalid_configs = [
        {
            "type": "nonexistent_evaluator"
        }
    ]
    
    with pytest.raises(ValueError):
        Experiment(
            name="test_experiment",
            dataset="test_dataset",
            task=task,
            evaluator_configs=invalid_configs
        ) 

"""
Tests for the evaluator registry.
"""

import pytest
from typing import Any

from arize.experimental.datasets.experiments.types import EvaluationResult

from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry


class TestEvaluator(BaseEvaluator):
    """Test evaluator for registry testing."""
    
    @property
    def name(self) -> str:
        return "test_evaluator"
    
    def evaluate(self, output: Any) -> EvaluationResult:
        return EvaluationResult(score=1.0, label="test")


def test_evaluator_registration():
    """Test registering an evaluator."""
    # Register a test evaluator
    EvaluatorRegistry.register("test_evaluator", TestEvaluator)
    
    # Verify it's registered
    assert "test_evaluator" in EvaluatorRegistry._evaluators
    assert EvaluatorRegistry.get("test_evaluator") == TestEvaluator


def test_evaluator_registration_decorator():
    """Test registering an evaluator using the decorator."""
    
    @EvaluatorRegistry.register("decorator_test")
    class DecoratorTestEvaluator(BaseEvaluator):
        @property
        def name(self) -> str:
            return "decorator_test"
        
        def evaluate(self, output: Any) -> EvaluationResult:
            return EvaluationResult(score=1.0, label="test")
    
    # Verify it's registered
    assert "decorator_test" in EvaluatorRegistry._evaluators
    assert EvaluatorRegistry.get("decorator_test") == DecoratorTestEvaluator


def test_duplicate_registration():
    """Test that registering the same name twice raises an error."""
    EvaluatorRegistry.register("duplicate_test", TestEvaluator)
    
    with pytest.raises(ValueError):
        EvaluatorRegistry.register("duplicate_test", TestEvaluator)


def test_get_nonexistent_evaluator():
    """Test that getting a nonexistent evaluator raises an error."""
    with pytest.raises(ValueError):
        EvaluatorRegistry.get("nonexistent_evaluator")


def test_list_evaluators():
    """Test listing registered evaluators."""
    # Register a few evaluators
    EvaluatorRegistry.register("list_test_1", TestEvaluator)
    EvaluatorRegistry.register("list_test_2", TestEvaluator)
    
    # Get the list
    evaluators = EvaluatorRegistry.list_evaluators()
    
    # Verify our test evaluators are in the list
    assert "list_test_1" in evaluators
    assert "list_test_2" in evaluators 

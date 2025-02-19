"""
Tests for the evaluator registry.
"""

from typing import Any

import pytest
from arize.experimental.datasets.experiments.types import EvaluationResult

from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry


@pytest.fixture
def test_evaluator_class():
    """Fixture providing a test evaluator class."""

    class TestEvaluator(BaseEvaluator):
        """Test evaluator for registry testing."""

        @property
        def name(self) -> str:
            return "test_evaluator"

        def evaluate(self, output: Any) -> EvaluationResult:
            return EvaluationResult(score=1.0, label="test")

        def __call__(self, output: Any) -> EvaluationResult:
            return self.evaluate(output)

    return TestEvaluator


def test_evaluator_registration(test_evaluator_class):
    """Test registering an evaluator."""
    # Clear registry before test
    EvaluatorRegistry._evaluators.clear()

    # Register a test evaluator
    EvaluatorRegistry.register("test_evaluator", test_evaluator_class)

    # Verify it's registered
    assert "test_evaluator" in EvaluatorRegistry._evaluators
    assert EvaluatorRegistry.get("test_evaluator") == test_evaluator_class


def test_evaluator_registration_decorator():
    """Test registering an evaluator using the decorator."""
    # Clear registry before test
    EvaluatorRegistry._evaluators.clear()

    @EvaluatorRegistry.register("decorator_test")
    class DecoratorTestEvaluator(BaseEvaluator):
        @property
        def name(self) -> str:
            return "decorator_test"

        def evaluate(self, output: Any) -> EvaluationResult:
            return EvaluationResult(score=1.0, label="test")

        def __call__(self, output: Any) -> EvaluationResult:
            return self.evaluate(output)

    # Verify it's registered
    assert "decorator_test" in EvaluatorRegistry._evaluators
    assert EvaluatorRegistry.get("decorator_test") == DecoratorTestEvaluator


def test_duplicate_registration(test_evaluator_class):
    """Test that registering the same name twice raises an error."""
    # Clear registry before test
    EvaluatorRegistry._evaluators.clear()

    EvaluatorRegistry.register("duplicate_test", test_evaluator_class)

    with pytest.raises(ValueError):
        EvaluatorRegistry.register("duplicate_test", test_evaluator_class)


def test_get_nonexistent_evaluator():
    """Test that getting a nonexistent evaluator raises an error."""
    # Clear registry before test
    EvaluatorRegistry._evaluators.clear()

    with pytest.raises(ValueError):
        EvaluatorRegistry.get("nonexistent_evaluator")


def test_list_evaluators(test_evaluator_class):
    """Test listing registered evaluators."""
    # Clear registry before test
    EvaluatorRegistry._evaluators.clear()

    # Register a few evaluators
    EvaluatorRegistry.register("list_test_1", test_evaluator_class)
    EvaluatorRegistry.register("list_test_2", test_evaluator_class)

    # Get the list
    evaluators = EvaluatorRegistry.list()

    # Verify our test evaluators are in the list
    assert "list_test_1" in evaluators
    assert "list_test_2" in evaluators

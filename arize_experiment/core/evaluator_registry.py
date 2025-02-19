"""
Core evaluator registry for managing and accessing evaluators.

This module provides a central registry for evaluators, allowing dynamic
registration and retrieval of evaluator classes.
"""

from typing import Callable, Dict, List, Type

from arize_experiment.core.evaluator import BaseEvaluator


class EvaluatorRegistry:
    """Central registry for evaluator classes.

    This class provides class methods for registering, retrieving, and listing
    available evaluators. It acts as a singleton registry accessible throughout
    the application.
    """

    _evaluators: Dict[str, Type[BaseEvaluator]] = {}

    @classmethod
    def register(
        cls, name: str, evaluator_class: Type[BaseEvaluator] | None = None
    ) -> Type[BaseEvaluator] | Callable[[Type[BaseEvaluator]], Type[BaseEvaluator]]:
        """Register an evaluator class with a unique name.

        Can be used as a decorator or called directly:

        @EvaluatorRegistry.register("my_evaluator")
        class MyEvaluator(BaseEvaluator):
            pass

        # OR

        EvaluatorRegistry.register("my_evaluator", MyEvaluator)

        Args:
            name: Unique identifier for the evaluator
            evaluator_class: Optional evaluator class to register

        Returns:
            Either the registered evaluator class (when called directly) or
            a decorator function (when used as a decorator)

        Raises:
            ValueError: If an evaluator is already registered with the given name
        """

        def decorator(cls: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
            if name in EvaluatorRegistry._evaluators:
                raise ValueError(f"Evaluator {name} is already registered")
            EvaluatorRegistry._evaluators[name] = cls
            return cls

        if evaluator_class is None:
            return decorator

        return decorator(evaluator_class)

    @classmethod
    def get(cls, name: str) -> Type[BaseEvaluator]:
        """Get an evaluator class by name.

        Args:
            name: The registered name of the evaluator

        Returns:
            The registered evaluator class

        Raises:
            ValueError: If no evaluator is registered with the given name
        """
        if name not in cls._evaluators:
            raise ValueError(f"No evaluator registered with name {name}")
        return cls._evaluators[name]

    @classmethod
    def list(cls) -> List[str]:
        """List all registered evaluator names.

        Returns:
            List of registered evaluator names
        """
        return list(cls._evaluators.keys())

"""
Core evaluator registry for managing and accessing evaluators.

This module provides a central registry for evaluators, allowing dynamic
registration and retrieval of evaluator classes.
"""

from typing import Dict, List, Type

from arize_experiment.core.evaluator import BaseEvaluator


class EvaluatorRegistry:
    """Central registry for evaluator classes.
    
    This class provides class methods for registering, retrieving, and listing
    available evaluators. It acts as a singleton registry accessible throughout
    the application.
    """
    
    _evaluators: Dict[str, Type[BaseEvaluator]] = {}
    
    @classmethod
    def register(cls, name: str, evaluator_class: Type[BaseEvaluator] = None):
        """Register an evaluator class with a unique name.
        
        Can be used as a decorator or called directly:
        
        @EvaluatorRegistry.register("my_evaluator")
        class MyEvaluator(BaseEvaluator):
            pass
            
        # OR
        
        EvaluatorRegistry.register("my_evaluator", MyEvaluator)
        
        Args:
            name: Unique identifier for the evaluator
            evaluator_class: The evaluator class to register (optional when used as decorator)
            
        Returns:
            The evaluator class (enables decorator usage)
            
        Raises:
            ValueError: If an evaluator is already registered with the given name
        """
        def _register(eval_class: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
            if name in cls._evaluators:
                raise ValueError(f"Evaluator already registered with name '{name}'")
            if not issubclass(eval_class, BaseEvaluator):
                raise ValueError(
                    f"Evaluator class must inherit from BaseEvaluator, got {eval_class}"
                )
            cls._evaluators[name] = eval_class
            return eval_class
        
        if evaluator_class is None:
            return _register
        return _register(evaluator_class)
    
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
            raise ValueError(f"No evaluator registered with name '{name}'")
        return cls._evaluators[name]
    
    @classmethod
    def list_evaluators(cls) -> List[str]:
        """List all registered evaluator names.
        
        Returns:
            List of registered evaluator names
        """
        return list(cls._evaluators.keys()) 

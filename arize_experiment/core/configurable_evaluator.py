"""
Configurable evaluator base class and configuration types.

This module provides the base infrastructure for creating evaluators
that can be configured through dictionaries.
"""

from dataclasses import dataclass
from typing import Any, Dict

from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry


@dataclass
class EvaluatorConfig:
    """Configuration for an evaluator.

    Attributes:
        type: The registered name of the evaluator
        params: Parameters to pass to the evaluator constructor
    """

    type: str
    params: Dict[str, Any]


class ConfigurableEvaluator(BaseEvaluator):
    """Base class for configurable evaluators.

    This class provides factory methods for creating evaluator instances
    from configuration dictionaries.
    """

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseEvaluator:
        """Create an evaluator instance from a config dict.

        Args:
            config: Dictionary containing evaluator configuration
                   Must include 'type' key and optional parameters

        Returns:
            Configured evaluator instance

        Raises:
            ValueError: If config is invalid or evaluator type not found
        """
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(config)}")

        if "type" not in config:
            raise ValueError("Config must include 'type' key")

        evaluator_type = config.pop("type")
        evaluator_class = EvaluatorRegistry.get(evaluator_type)

        try:
            return evaluator_class(**config)
        except TypeError as e:
            raise ValueError(f"Invalid parameters for evaluator {evaluator_type}: {e}")
        finally:
            # Restore the type key in case config dict is reused
            config["type"] = evaluator_type

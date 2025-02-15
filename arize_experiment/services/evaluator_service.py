"""
Service for managing and accessing evaluators.
"""

import logging
from typing import Dict, List, Optional, Type

from arize_experiment.core.evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class EvaluatorServiceError(Exception):
    """Raised when there are issues with the evaluator service."""
    pass


class EvaluatorService:
    """Service for managing evaluators.
    
    This service handles:
    1. Registering new evaluator types
    2. Creating evaluator instances
    3. Looking up evaluators by name
    4. Validating evaluator configurations
    """

    def __init__(self):
        """Initialize the evaluator service."""
        self._evaluator_types: Dict[str, Type[BaseEvaluator]] = {}
        self._instances: Dict[str, BaseEvaluator] = {}

    def register_type(self, evaluator_class: Type[BaseEvaluator]) -> None:
        """Register a new evaluator type.
        
        Args:
            evaluator_class: The evaluator class to register
        
        Raises:
            EvaluatorServiceError: If registration fails
        """
        try:
            # Create temporary instance to get name
            temp_instance = evaluator_class()
            name = temp_instance.name

            if name in self._evaluator_types:
                raise EvaluatorServiceError(
                    f"Evaluator type '{name}' is already registered"
                )

            logger.info(f"Registering evaluator type: {name}")
            self._evaluator_types[name] = evaluator_class

        except Exception as e:
            error_msg = f"Failed to register evaluator type: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EvaluatorServiceError(error_msg) from e

    def create_evaluator(self, name: str, **kwargs) -> BaseEvaluator:
        """Create a new evaluator instance.
        
        Args:
            name: Name of the evaluator type to create
            **kwargs: Configuration parameters for the evaluator
        
        Returns:
            New evaluator instance
        
        Raises:
            EvaluatorServiceError: If creation fails
        """
        try:
            if name not in self._evaluator_types:
                raise EvaluatorServiceError(f"Unknown evaluator type: {name}")

            logger.debug(f"Creating evaluator instance: {name}")
            evaluator_class = self._evaluator_types[name]
            instance = evaluator_class(**kwargs)

            # Validate the instance
            if not instance.validate():
                raise EvaluatorServiceError(
                    f"Evaluator validation failed: {name}"
                )

            # Cache the instance
            instance_key = self._get_instance_key(name, kwargs)
            self._instances[instance_key] = instance

            return instance

        except Exception as e:
            error_msg = f"Failed to create evaluator: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EvaluatorServiceError(error_msg) from e

    def get_evaluator(
        self, name: str, **kwargs
    ) -> Optional[BaseEvaluator]:
        """Get an existing evaluator instance or create a new one.
        
        Args:
            name: Name of the evaluator type
            **kwargs: Configuration parameters for the evaluator
        
        Returns:
            Existing or new evaluator instance, or None if type not found
        """
        instance_key = self._get_instance_key(name, kwargs)
        
        # Return cached instance if it exists
        if instance_key in self._instances:
            return self._instances[instance_key]

        # Create new instance if type exists
        if name in self._evaluator_types:
            return self.create_evaluator(name, **kwargs)

        return None

    def list_evaluator_types(self) -> List[str]:
        """Get list of registered evaluator type names.
        
        Returns:
            List of evaluator type names
        """
        return list(self._evaluator_types.keys())

    def _get_instance_key(self, name: str, config: dict) -> str:
        """Generate a unique key for an evaluator instance.
        
        This combines the evaluator name and its configuration to create
        a unique identifier for caching purposes.
        
        Args:
            name: Evaluator type name
            config: Evaluator configuration
        
        Returns:
            Unique instance key
        """
        # Sort config items for consistent key generation
        config_str = str(sorted(config.items()))
        return f"{name}:{config_str}"

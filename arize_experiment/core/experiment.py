"""
Core experiment domain model for arize-experiment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from arize_experiment.core.configurable_evaluator import ConfigurableEvaluator
from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.task import Task, TaskResult


@dataclass
class ExperimentResult:
    """Result of an experiment run."""

    task_result: TaskResult
    evaluations: List[Tuple[float, str, Optional[str]]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if the experiment ran successfully."""
        return self.error is None and self.task_result.success


@dataclass
class Experiment:
    """Core experiment domain model.

    An experiment consists of a task to run and evaluators to assess
    the task's output. Evaluators can be provided either as instances
    or as configuration dictionaries.
    """

    name: str
    dataset: str
    task: Task
    evaluator_configs: List[Dict[str, Any]]
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    _evaluators: List[BaseEvaluator] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Initialize and validate the experiment configuration."""
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
        if not self.dataset:
            raise ValueError("Dataset name cannot be empty")
        if not isinstance(self.task, Task):
            raise ValueError(f"Task must be an instance of Task, got {type(self.task)}")
        
        # Initialize evaluators from configs
        self._evaluators = []
        for config in self.evaluator_configs:
            try:
                evaluator = ConfigurableEvaluator.from_config(config)
                self._evaluators.append(evaluator)
            except ValueError as e:
                raise ValueError(f"Invalid evaluator configuration: {e}")

    @property
    def evaluators(self) -> List[BaseEvaluator]:
        """Get the list of initialized evaluators."""
        return self._evaluators

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary format.

        This is useful for serialization and API communication.

        Returns:
            Dict containing the experiment configuration
        """
        return {
            "name": self.name,
            "dataset": self.dataset,
            "task": self.task.name,
            "evaluator_configs": self.evaluator_configs,
            "description": self.description,
            "tags": self.tags,
        }

    def __str__(self) -> str:
        """Get a string representation of the experiment."""
        return (
            f"Experiment(name={self.name}, dataset={self.dataset}, "
            f"task={self.task.name}, evaluators={[e.name for e in self.evaluators]})"
        )

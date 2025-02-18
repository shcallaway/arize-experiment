"""
Core experiment domain model for arize-experiment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

    An experiment consists of a task to run and optional evaluators to assess
    the task's output. The experiment can be configured with metadata tags
    and a description.
    """

    name: str
    dataset: str
    task: Task
    evaluators: List[BaseEvaluator]
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the experiment configuration."""
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
        if not self.dataset:
            raise ValueError("Dataset name cannot be empty")
        if not isinstance(self.task, Task):
            raise ValueError(f"Task must be an instance of Task, got {type(self.task)}")
        if not all(isinstance(e, BaseEvaluator) for e in self.evaluators):
            raise ValueError("All evaluators must be instances of BaseEvaluator")

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
            "evaluators": [e.name for e in self.evaluators],
            "description": self.description,
            "tags": self.tags,
        }

    def __str__(self) -> str:
        """Get a string representation of the experiment."""
        return (
            f"Experiment(name={self.name}, dataset={self.dataset}, "
            f"task={self.task.name}, evaluators={[e.name for e in self.evaluators]})"
        )

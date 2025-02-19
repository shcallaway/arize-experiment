"""
Core task registry for managing and accessing tasks.

This module provides a central registry for tasks, allowing dynamic
registration and retrieval of task classes.
"""

from typing import Callable, Dict, List, Type

from arize_experiment.core.task import Task


class TaskRegistry:
    """Central registry for task classes.

    This class provides class methods for registering, retrieving, and listing
    available tasks. It acts as a singleton registry accessible throughout
    the application.
    """

    _tasks: Dict[str, Type[Task]] = {}

    @classmethod
    def register(
        cls, name: str, task_class: Type[Task] | None = None
    ) -> Callable[[Type[Task]], Type[Task]] | Type[Task]:
        """Register a task class with a unique name.

        Can be used as a decorator or called directly:

        @TaskRegistry.register("my_task")
        class MyTask(Task):
            pass

        # OR

        TaskRegistry.register("my_task", MyTask)

        Args:
            name: Unique identifier for the task
            task_class: Optional task class to register

        Returns:
            Either the registered task class (when called directly) or
            a decorator function (when used as a decorator)

        Raises:
            ValueError: If a task is already registered with the given name
        """

        def decorator(cls: Type[Task]) -> Type[Task]:
            if name in TaskRegistry._tasks:
                raise ValueError(f"Task '{name}' is already registered")
            TaskRegistry._tasks[name] = cls
            return cls

        if task_class is not None:
            return decorator(task_class)
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[Task]:
        """Get a task class by name.

        Args:
            name: Name of the task to retrieve

        Returns:
            The task class

        Raises:
            ValueError: If no task is registered with the given name
        """
        if name not in cls._tasks:
            raise ValueError(f"No task registered with name '{name}'")
        return cls._tasks[name]

    @classmethod
    def list(cls) -> List[str]:
        """List all registered task names.

        Returns:
            List of registered task names
        """
        return list(cls._tasks.keys())

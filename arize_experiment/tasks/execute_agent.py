"""Execute an agent by calling a web server."""

from arize_experiment.core.task import Task, TaskResult


class ExecuteAgentTask(Task):
    """Execute an agent by calling a web server."""

    def __init__(self, url: str):
        self.url = url

    @property
    def name(self) -> str:
        """Get the task name."""
        return "execute_agent"

    def execute(self, Input: str) -> TaskResult:
        """Execute the agent on input text.

        Args:
            Input: Single text string

        Returns:
            TaskResult containing:
                output: TODO
                metadata: Processing information including model used
                error: Any error message if task failed
        """
        return TaskResult(
            output=None,
            metadata={}
        )

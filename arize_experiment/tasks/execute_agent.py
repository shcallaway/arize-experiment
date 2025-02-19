"""Execute an agent by calling a web server."""

from arize_experiment.core.task import Task, TaskResult


class ExecuteAgentTask(Task):
    """Execute an agent by calling a web server.
    
    This task makes HTTP requests to a specified endpoint to execute an agent
    and process its response. The agent is expected to be running at the
    provided URL endpoint.
    """

    def __init__(self, url: str):
        """Initialize the execute agent task.
        
        Args:
            url: The endpoint URL where the agent is running
        """
        self.url = url

    @property
    def name(self) -> str:
        """Get the task name.
        
        Returns:
            str: The unique identifier for this task
        """
        return "execute_agent"

    def execute(self, Input: str) -> TaskResult:
        """Execute the agent on input text.

        Args:
            Input: Single text string to be processed by the agent

        Returns:
            TaskResult containing:
                output: The agent's response (currently unimplemented)
                metadata: Processing information including request details
                error: Any error message if the request or processing failed
        
        Raises:
            TaskError: If the HTTP request fails or agent processing fails
        """
        return TaskResult(output=None, metadata={"url": self.url})

"""
Service for running experiments and managing their execution.
"""

import logging
from typing import List, Optional

from arize_experiment.core.experiment import Experiment, ExperimentResult
from arize_experiment.core.evaluator import EvaluationResult
from arize_experiment.core.task import TaskResult
from arize_experiment.infrastructure.arize_client import ArizeClient

logger = logging.getLogger(__name__)


class ExperimentRunnerError(Exception):
    """Raised when there are issues running an experiment."""
    pass


class ExperimentRunner:
    """Service for running experiments.
    
    This service handles:
    1. Validating experiment configuration
    2. Executing the experiment's task
    3. Running evaluators on the task output
    4. Collecting and returning results
    """

    def __init__(self, arize_client: 'ArizeClient'):  # Forward reference for type hint
        """Initialize the runner with required dependencies.
        
        Args:
            arize_client: Client for interacting with Arize API
        """
        self.client = arize_client

    def run(self, experiment: Experiment, input_data: Optional[any] = None) -> ExperimentResult:
        """Run an experiment and return its results.
        
        This method:
        1. Validates the experiment configuration
        2. Creates an experiment on Arize using the client
        3. Returns the results
        
        Args:
            experiment: The experiment to run
            input_data: Optional input data for the task
        
        Returns:
            ExperimentResult containing task output and evaluation results
        
        Raises:
            ExperimentRunnerError: If there are issues running the experiment
        """
        try:
            # Validate experiment configuration
            logger.debug(f"Validating experiment: {experiment}")
            experiment.validate()

            # Convert task and evaluators to callables for Arize API
            task_fn = lambda input: experiment.task.execute(input).output
            evaluator_fns = [
                lambda output, e=evaluator: e.evaluate(output).score
                for evaluator in experiment.evaluators
            ]

            # Run experiment using Arize client
            logger.info(f"Running experiment on Arize: {experiment.name}")
            arize_result = self.client.run_experiment(
                experiment_name=experiment.name,
                dataset_name=experiment.dataset,
                task=task_fn,
                evaluators=evaluator_fns if experiment.evaluators else None
            )

            # Convert Arize result to ExperimentResult
            task_result = TaskResult(output=arize_result, error=None)
            evaluation_results = [
                EvaluationResult(
                    score=score,
                    label="arize_evaluation",
                    explanation="Evaluation performed on Arize platform"
                )
                for score in (arize_result.get("scores", []) if isinstance(arize_result, dict) else [])
            ]

            return ExperimentResult(
                task_result=task_result,
                evaluations=evaluation_results
            )

        except Exception as e:
            error_msg = f"Failed to run experiment: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ExperimentRunnerError(error_msg) from e

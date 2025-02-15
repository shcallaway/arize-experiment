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
            def task_fn(input_data):
                # Convert Arize types to string if needed
                if hasattr(input_data, '_asdict'):
                    input_data = str(input_data._asdict())
                elif not isinstance(input_data, str):
                    input_data = str(input_data)
                
                result = experiment.task.execute(input_data)
                # For evaluation, return just the task output
                # But include metadata for Arize API
                if result.error:
                    return {
                        "id": experiment.name,
                        "error": result.error
                    }
                return result.output
            
            def create_evaluator_fn(evaluator):
                def evaluator_fn(output):
                    # Convert Arize types to dict if needed
                    if hasattr(output, '_asdict'):
                        output = output._asdict()
                    return evaluator.evaluate(output).score
                return evaluator_fn
            
            evaluator_fns = [
                create_evaluator_fn(evaluator)
                for evaluator in experiment.evaluators
            ] if experiment.evaluators else None

            # Run experiment using Arize client
            logger.info(f"Running experiment on Arize: {experiment.name}")
            try:
                arize_result = self.client.run_experiment(
                    experiment_name=experiment.name,
                    dataset_name=experiment.dataset,
                    task=task_fn,
                    evaluators=evaluator_fns if experiment.evaluators else None
                )
            except Exception as e:
                if "ArrowKeyError" in str(type(e)) and "experiment does not exist" in str(e).lower():
                    # Experiment doesn't exist yet, try to create it
                    logger.info(f"Creating new experiment: {experiment.name}")
                    arize_result = self.client.run_experiment(
                        experiment_name=experiment.name,
                        dataset_name=experiment.dataset,
                        task=task_fn,
                        evaluators=evaluator_fns if experiment.evaluators else None
                    )
                else:
                    raise

            # Ensure we have a valid result structure
            if not arize_result or not isinstance(arize_result, dict):
                raise ExperimentRunnerError("Invalid result format from Arize API")

            # Extract run ID and results
            run_id = arize_result.get("id")
            if not run_id:
                raise ExperimentRunnerError("Missing run ID in Arize API response")

            # Get the actual task output
            output = arize_result.get("output")
            
            # Handle various output types
            if hasattr(output, '_asdict'):
                output = output._asdict()
            elif output is None:
                # If no specific output, use the full result minus internal fields
                output = {k: v for k, v in arize_result.items() if k not in ["id", "scores"]}
            
            # Get error if present
            error = arize_result.get("error")
            task_result = TaskResult(output=output, error=error)
            
            # Handle evaluation results
            evaluation_results = []
            if "scores" in arize_result:
                evaluation_results = [
                    EvaluationResult(
                        score=score,
                        label="arize_evaluation",
                        explanation="Evaluation performed on Arize platform"
                    )
                    for score in arize_result["scores"]
                ]

            return ExperimentResult(
                task_result=task_result,
                evaluations=evaluation_results
            )

        except Exception as e:
            error_msg = f"Failed to run experiment: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ExperimentRunnerError(error_msg) from e

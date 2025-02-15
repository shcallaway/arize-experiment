"""
arize-experiment: A tool for running experiments on Arize.
"""

__version__ = "0.2.0"  # Updated version to reflect major refactor

from arize_experiment.cli.commands import main
from arize_experiment.experiments import (
    Experiment,
    ExperimentError,
    DatasetExperiment,
)
from arize_experiment.config import (
    ArizeConfig,
    ExperimentConfig,
    create_experiment_config,
)

__all__ = [
    "__version__",
    "main",
    "Experiment",
    "ExperimentError",
    "DatasetExperiment",
    "ArizeConfig",
    "ExperimentConfig",
    "create_experiment_config",
]

# Arize Experiment

A powerful CLI tool for evaluating AI model performance through systematic experiments. Run standardized evaluations of language models and AI agents, collect metrics, and analyze results using Arize's analytics platform.

## Project Status

**Current Status**: Beta

- Active development with regular updates
- Core features stable and production-ready
- Actively seeking community feedback and contributions

**Roadmap**:

- Support for additional LLM providers beyond OpenAI and Anthropic
- Enhanced metric collection and visualization
- Custom evaluation pipeline builder
- Support for pluggable evaluators and tasks

## Getting Started

### Prerequisites

- Python 3.10.13 (Other versions including Python 3.11+ not currently supported)
- Git
- OpenAI API key (for OpenAI-based tasks)
- Anthropic API key (for Claude-based tasks)
- Arize account and API credentials

### Clone the Repository

```bash
git clone https://github.com/Arize-ai/arize-experiment.git
cd arize-experiment
```

### Python Setup

```bash
# Install pyenv
brew install pyenv

# Configure shell for pyenv (add to ~/.zshrc or ~/.bashrc)
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install required Python version
pyenv install 3.10.13
pyenv local 3.10.13

# Create virtual environment
python -m venv venv
source venv/bin/activate
```

### Package Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the arize-experiment package in editable mode
pip install -e .
```

### Environment Configuration

Create your environment file:

```bash
cp .env.example .env
```

Required environment variables:

```bash
# Arize API Credentials
# Required for all tasks/evaluators
# Obtain these from https://app.arize.com/settings/api
ARIZE_API_KEY=your_arize_api_key
ARIZE_SPACE_KEY=your_arize_space_key

# OpenAI API Configuration
# Required for OpenAI-based tasks/evaluators
OPENAI_API_KEY=your_openai_api_key

# Chatbot Server Configuration
# Required for call_chatbot_server task
CHATBOT_SERVER_URL=http://localhost:8080
```

To obtain Arize credentials:

1. Sign up at [Arize AI Platform](https://app.arize.com)
2. Navigate to Settings → API Keys
3. Create a new API key and copy both the API key and space key

## Features

- **Standardized Evaluation Framework**: Run systematic evaluations of AI models with consistent metrics
- **Comprehensive Analytics**: Track and compare performance across experiments
- **Flexible Task System**: Support for multiple evaluation tasks and metrics
- **Arize Integration**: Automatic upload of results to Arize's analytics platform

Currently supported capabilities:

- **Tasks**
  - Classify Sentiment: Classify input text as positive, negative, or neutral
  - Call Chatbot Server: Make an API request to an instance of a chatbot server
  - Delegate: Make an API request to another service that will handle the task
  - Echo: Return the input as it was received
- **Evaluators**
  - Chatbot Response is Acceptable: Measure the quality of the response from the chatbot server
  - Sentiment Classification is Accurate: Measure the accuracy of the sentiment classifier task output

## Dataset Requirements

### Dataset Format

Each task type requires a specific dataset format. Here are the requirements for each:

#### Classify Sentiment Dataset

Columns:

- `input`: The text content to analyze

#### Chatbot Server Dataset

Columns:

- `input`: A JSON object containing the conversation history between the user and the chatbot

`input` example:

```json
[
  {
    "role": "user",
    "content": "Hello"
  },
  {
    "role": "assistant",
    "content": "Hi there!"
  },
  {
    "role": "user",
    "content": "What's the weather?"
  }
]
```

### Creating Custom Datasets

1. Create a CSV file with your dataset
2. Use the `arize-experiment create-dataset` command to upload your dataset to Arize

```bash
arize-experiment create-dataset \
  --name <dataset-name> \
  --path-to-csv <path-to-csv-file>
```

## Usage Guide

### Basic Command Structure

```bash
arize-experiment run \
  --name <experiment-name> \
  --dataset <dataset-name> \
  --task <task-name> \
  --evaluator <evaluator-name>
```

### Tasks

#### 1. Classify Sentiment

Analyzes text sentiment using LLMs. Classifies text as positive, negative, or neutral.

```bash
# Example: Evaluate sentiment classification
arize-experiment run \
  --name my-experiment \
  --dataset my-dataset \
  --task classify_sentiment \
  --evaluator sentiment_classification_is_accurate
```

#### 2. Call Chatbot Server

Calls a chatbot server by making HTTP requests to a specified endpoint.

```bash
# Example: Evaluate chatbot responses
arize-experiment run \
  --name my-experiment \
  --dataset my-dataset \
  --task call_chatbot_server \
  --evaluator chatbot_response_is_acceptable
```

## Development Guide

### Project Structure

```
arize_experiment/
├── cli/                          # Command-line interface
│   ├── main.py                  # CLI entry point
│   └── handler.py               # Command handlers
├── core/                        # Core functionality
│   ├── task.py                 # Base task class
│   ├── evaluator.py            # Base evaluator class
│   └── metrics.py              # Metric collection
├── evaluators/                  # Evaluation implementations
├── tasks/                       # Task implementations
└── __main__.py                 # Package entry point
```

### Dependencies

The project relies on several key packages:

- **Core Dependencies**

  - `arize[Datasets]>=7.0.0`: Arize AI platform integration
  - `openai>=1.0.0`: OpenAI API client
  - `anthropic>=0.19.1`: Anthropic API client
  - `pandas>=2.0.0`: Data manipulation and analysis
  - `python-dotenv>=1.0.0`: Environment variable management
  - `click>=8.0.0`: CLI framework
  - `urllib3>=2.0.0`: HTTP client library

- **Development Dependencies**
  - `pytest>=7.0.0`: Testing framework
  - `flake8>=7.0.0`: Code linting
  - `black>=24.1.0`: Code formatting
  - `mypy>=1.8.0`: Static type checking
  - `pandas-stubs>=2.2.0`: Type stubs for pandas

### Creating New Tasks

1. Create a new file in `tasks/` directory:

```python
# tasks/my_custom_task.py
from arize_experiment.core.task import Task, TaskResult
from arize_experiment.core.schema import DatasetSchema, ColumnSchema, DataType
from arize_experiment.core.task_registry import TaskRegistry
from typing import Dict, Any

@TaskRegistry.register("my_custom_task")  # Register the task with the framework
class MyCustomTask(Task):
    """A custom task implementation that processes input text.

    This task demonstrates the basic structure required for creating new tasks
    in the arize-experiment framework. It inherits from the base Task class
    and implements all required abstract methods.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the task with configuration parameters.

        Args:
            config: Dictionary containing task-specific configuration parameters
                   that will be used during execution.
        """
        super().__init__()  # Required call to parent class initializer
        self.config = config

    @property
    def name(self) -> str:
        """Define a unique identifier for this task.

        Returns:
            str: A lowercase string with underscores that uniquely identifies
                 this task type in the framework.
        """
        return "my_custom_task"

    @property
    def required_schema(self) -> DatasetSchema:
        """Define the expected structure of input data for this task.

        This schema is used to validate input data before execution.
        In this example, we require a single 'input' column of type string.

        Returns:
            DatasetSchema: Schema object describing required input format
        """
        return DatasetSchema(
            columns={
                "input": ColumnSchema(
                    name="input",
                    types=[DataType.STRING],  # Accepts string data only
                    required=True  # This field must be present
                )
            }
        )

    def execute(self, dataset_row: Dict[str, Any]) -> TaskResult:
        """Execute the task's core logic on a single input row.

        This method contains the main processing logic for the task.
        It handles errors gracefully and returns results in a standardized format.

        Args:
            dataset_row: A dictionary containing the input data matching the
                        required_schema structure.

        Returns:
            TaskResult: Object containing:
                - dataset_row: The original input data
                - output: The processed result (or None if error)
                - metadata: Additional execution information
                - error: Error message if processing failed
        """
        try:
            # Process the input using a helper method (not shown)
            result = self._process_input(dataset_row["input"])

            # Return successful result with metadata
            return TaskResult(
                dataset_row=dataset_row,
                output=result,
                metadata={"config": self.config}  # Include config for tracking
            )
        except Exception as e:
            # Return error result while preserving the input
            return TaskResult(
                dataset_row=dataset_row,
                output=None,  # No output on error
                error=str(e)  # Convert exception to string message
            )

# Alternative registration method:
# TaskRegistry.register("my_custom_task", MyCustomTask)
```

### Creating New Evaluators

1. Create a new file in `evaluators/` directory:

```python
# evaluators/my_custom_evaluator.py
from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.task import TaskResult
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize.experimental.datasets.experiments.types import EvaluationResult
from typing import Dict, Any

@EvaluatorRegistry.register("my_custom_evaluator")  # Register the evaluator with the framework
class MyCustomEvaluator(BaseEvaluator):
    """An evaluator that assesses task output quality using a threshold.

    This evaluator demonstrates the basic structure required for creating
    new evaluators in the arize-experiment framework. It inherits from
    BaseEvaluator and implements all required abstract methods.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        """Initialize the evaluator with a quality threshold.

        Args:
            threshold: A float between 0 and 1 representing the minimum
                      acceptable score for the evaluation to pass.
                      Defaults to 0.8 (80%).
        """
        super().__init__()  # Required call to parent class initializer
        self.threshold = threshold

    @property
    def name(self) -> str:
        """Define a unique identifier for this evaluator.

        Returns:
            str: A lowercase string with underscores that uniquely identifies
                 this evaluator type in the framework.
        """
        return "my_custom_evaluator"

    def evaluate(self, task_result: TaskResult) -> EvaluationResult:
        """Evaluate the quality of a task's output.

        This method implements the core evaluation logic. It takes a task's
        output and returns a standardized evaluation result with a score
        and pass/fail determination.

        Args:
            task_result: The complete result from a task execution, including
                        input data, output, and any metadata or errors.

        Returns:
            EvaluationResult: Object containing:
                - score: A float between 0 and 1 indicating quality
                - passed: Boolean indicating if score meets threshold
                - metadata: Additional evaluation context
                - explanation: Human-readable description of the result
        """
        # Calculate quality score using helper method (not shown)
        score = self._calculate_score(task_result)

        # Return standardized evaluation result
        return EvaluationResult(
            score=score,  # Quality score between 0 and 1
            passed=score >= self.threshold,  # Pass if score meets threshold
            metadata={"threshold": self.threshold},  # Include config for tracking
            explanation=f"Score {score} {'meets' if score >= self.threshold else 'does not meet'} threshold {self.threshold}"
        )

# Alternative registration method:
# EvaluatorRegistry.register("my_custom_evaluator", MyCustomEvaluator)
```

Note: The framework uses a registry system to manage tasks and evaluators. You can register your implementations either using the decorator syntax shown above or by calling the register method directly. The registration makes your task/evaluator available to the CLI and other framework components.

### Testing Guidelines

We use pytest for testing. All tests should be placed in the `tests/` directory.

#### Test Structure

```python
# tests/test_my_custom_task.py
import pytest
from tasks.my_custom_task import MyCustomTask

def test_my_custom_task_basic():
    task = MyCustomTask({})
    input_data = {"test": "data"}
    result = task.execute(input_data)
    assert result is not None

def test_my_custom_task_validation():
    task = MyCustomTask({})
    invalid_input = {}
    with pytest.raises(ValueError):
        task.validate_input(invalid_input)
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_my_custom_task.py

# Run with coverage
pytest --cov=arize_experiment tests/

# Run with verbose output
pytest -v
```

### Best Practices & Performance Considerations

#### API Rate Limits

Please be aware of the rate limits for external services:

- OpenAI API: Varies by tier and model
- Anthropic API: Varies by tier and model
- Arize API: Please refer to your service agreement

## Troubleshooting Guide

### Common Issues

1. **Python Not Found After Installation**

   ```bash
   pyenv rehash
   eval "$(pyenv init -)"
   ```

2. **Environment Variables Not Loading**

   - Check file permissions: `chmod 600 .env`
   - Verify file location: Must be in project root
   - Use `printenv` to verify variables are set

3. **Dataset Loading Errors**

   - Verify JSON format matches requirements
   - Check file encoding (use UTF-8)
   - Ensure all required fields are present

4. **Task Execution Failures**
   - Check API key validity
   - Verify network connectivity
   - Ensure input data matches schema
   - Check API service status

### Development Tools

The project uses several development tools:

- **Black**: Code formatting

  ```bash
  black arize_experiment/
  ```

- **Flake8**: Code linting

  ```bash
  flake8 arize_experiment/
  ```

- **MyPy**: Type checking

  ```bash
  mypy arize_experiment/
  ```

- **Pre-commit**: Git hooks
  ```bash
  pre-commit install
  pre-commit run --all-files
  ```

## Contributing

### Getting Started

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Document all public methods
- Keep functions focused and small

### Community

- Report issues on GitHub
- Contribute to discussions
- Share your use-cases

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3). This means:

- You can use this software for any purpose
- You can modify and distribute this software
- If you distribute modified versions, they must also be under GPLv3
- All changes must be documented and source code must be available

See the [LICENSE](LICENSE) file for the complete license text.

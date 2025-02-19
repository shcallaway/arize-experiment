# Arize Experiment

A powerful CLI tool for evaluating AI model performance through systematic experiments. Run standardized evaluations of language models and AI agents, collect metrics, and analyze results using Arize's analytics platform.

## Quick Start

Get up and running in under 5 minutes:

```bash
# Install Python 3.10 and create virtual environment
pyenv install 3.10.13
pyenv local 3.10.13

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Create an env file and add your secret values
cp .env.example .env

# Run your first experiment
arize-experiment run \
  --name <your-experiment-name> \
  --dataset <your-dataset-name> \
  --task <your-task-name> \
  --evaluator <your-evaluator-name>
```

## Features

- **Standardized Evaluation Framework**: Run systematic evaluations of AI models with consistent metrics
- **Comprehensive Analytics**: Track and compare performance across experiments
- **Flexible Task System**: Support for multiple evaluation tasks and metrics
- **Arize Integration**: Automatic upload of results to Arize's analytics platform

Currently supported capabilities:

- **Tasks**
  - Sentiment Classification: Evaluate text sentiment analysis
  - Agent Execution: Test AI agent performance
- **Evaluators**
  - Sentiment Classification Accuracy: Measure classification performance

## Installation & Setup

### System Requirements

- Python 3.10 (Python 3.11+ not currently supported)
- OpenAI API key (for LLM-based tasks)
- Arize account and API credentials

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

# Install the arize-experiment package
pip install -e .
```

### Environment Configuration

```bash
# Copy example env file
cp .env.example .env

# Set required variables in your .env:
ARIZE_API_KEY=your_arize_api_key
ARIZE_SPACE_KEY=your_arize_space_key
OPENAI_API_KEY=your_openai_api_key  # Required for certain tasks/evaluators
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

#### 1. Sentiment Classification

Analyzes text sentiment using OpenAI models. Classifies text as positive, negative, or neutral.

```bash
# Example: Evaluate a task that performs sentiment classification
arize-experiment run \
  --name <your-experiment-name> \
  --dataset <your-dataset-name> \
  --task sentiment_classification \
  --evaluator sentiment_classification_accuracy
```

#### 2. Agent Execution

Tests AI agent performance in predefined scenarios.

```bash
# Example: Evaluate a task that executes a chatbot
arize-experiment run \
  --name <your-experiment-name> \
  --dataset <your-dataset-name> \
  --task execute_agent \
  --evaluator agent_response_quality
```

### Working with Datasets

This script requires you to provide a dataset that matches the task you're evaluating. If you don't have a dataset, log into the Arize dashboard to create one.

### Viewing Results

Results are automatically uploaded to your Arize dashboard, where you can:

- View accuracy metrics and confusion matrices
- Compare experiments and model versions
- Generate performance reports
- Export results for further analysis

## Troubleshooting & FAQ

### Common Issues

1. **Python Not Found After Installation**

   ```bash
   pyenv rehash
   eval "$(pyenv init -)"
   ```

2. **Environment Variables Not Loading**

   - Ensure `.env` file is in the correct directory
   - Check file permissions
   - Verify variable names match exactly

3. **Dataset Loading Errors**

   - Verify dataset with this name exists in the Arize dashboard
   - Verify dataset columns match expected columns in task and evaluator

### Solutions

- Run with `LOG_LEVEL=DEBUG` for detailed logging
- Check Arize dashboard for experiment status
- Verify Arize API credentials are correct
- Ensure Python version is exactly 3.10.x

## Development Guide

### Project Structure

```
arize_experiment/
├── cli/            # Command-line interface
├── core/           # Core functionality
├── evaluators/     # Evaluation implementations
├── tasks/          # Task implementations
└── tests/          # Test suite
```

### Adding New Tasks/Evaluators

1. Create new task in `tasks/`
2. Implement evaluator in `evaluators/`
3. Register new evaluators and tasks in `cli/handler.py`
4. Add tests in `tests/`

### Testing Guidelines

Run tests:

```bash
pytest
```

Run tests for specific file:

```bash
pytest tests/<filename>.py
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

# Arize Experiment

A CLI tool for creating and running experiments to evaluate AI model performance using the Arize platform. This tool allows you to run standardized evaluations of language models and AI agents, collect metrics, and analyze results through Arize's analytics platform.

## Overview

Arize Experiment helps you:

- Run systematic evaluations of AI models
- Collect standardized metrics across different model versions
- Track and compare performance across experiments
- Generate detailed reports and analytics

Currently supported capabilities:

- **Tasks**:
  - Sentiment Classification: Evaluate text sentiment analysis
  - Agent Execution: Test and evaluate AI agent performance
- **Evaluators**:
  - Sentiment Classification Accuracy: Measure accuracy of sentiment classification

## Prerequisites

- Python 3.10 (Python 3.11+ not supported due to dependency constraints)
- pyenv (recommended for Python version management)
- OpenAI API key (for running LLM-based tasks)
- Arize account and API credentials

## pyenv

pyenv is a Python version manager that lets you easily switch between multiple versions of Python. It's similar to nvm for Node.js.

### pyenv Commands

Common pyenv commands you'll use:

```bash
# List all available Python versions
pyenv install --list

# List installed versions
pyenv versions

# Install a specific version
pyenv install 3.10.13

# Set global Python version
pyenv global 3.10.13

# Set local version (creates .python-version file)
pyenv local 3.10.13

# Show current Python version
pyenv version

# Uninstall a version
pyenv uninstall 3.10.13
```

### pyenv Configuration

pyenv needs to be properly configured in your shell to work. Add these lines to your shell configuration file (e.g., `~/.zshrc` or `~/.bashrc`):

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

After modifying your shell configuration, reload it:

```bash
source ~/.zshrc  # or source ~/.bashrc
```

### Common Issues

1. **Python command not found**

   ```bash
   pyenv rehash
   eval "$(pyenv init -)"
   ```

2. **python-build: definition not found**

   ```bash
   pyenv update
   ```

## Setup

### 1. Install and Configure pyenv

```bash
# Install pyenv
brew install pyenv

# Add to your shell (add to ~/.zshrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Reload shell
source ~/.zshrc
```

### 2. Install Python 3.10

```bash
pyenv install 3.10.13
pyenv local 3.10.13
python --version  # Should show Python 3.10.13
```

### 3. Set up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install Package

```bash
pip install -e .
```

## Configuration

Create a `.env` file by copying the `.env.example` file:

```bash
cp .env.example .env
```

Required environment variables:

```bash
ARIZE_API_KEY=your_arize_api_key
ARIZE_SPACE_KEY=your_arize_space_key

# Required for certain tasks and evaluators, but not all
OPENAI_API_KEY=your_openai_api_key
```

You can find the Arize credentials in your Arize account settings. The OpenAI API key can be obtained from the OpenAI platform.

## Usage

### Basic Command Structure

```bash
arize-experiment run \
  --name <experiment-name> \
  --dataset <dataset-name> \
  --task <task-name> \
  --evaluator <evaluator-name>
```

### Required Flags

- `--name`, `-n`: Experiment name
- `--dataset`, `-d`: Dataset name
- `--task`, `-t`: Task to execute (see available tasks below)
- `--evaluator`, `-e`: Evaluator(s) to use (can specify multiple)

### Optional Flags

- `--tag`: Add key=value tags to your experiment (can specify multiple)

### Available Tasks

1. **Sentiment Classification**

   Analyzes text to determine its emotional tone. The task uses OpenAI's models to classify text into three categories: positive, negative, or neutral. Ideal for:

   - Customer feedback analysis
   - Social media sentiment tracking
   - Product review evaluation

   Example usage:

   ```bash
   arize-experiment run \
     --name sentiment-test \
     --dataset customer-feedback \
     --task sentiment_classification \
     --evaluator sentiment_classification_accuracy
   ```

2. **Agent Execution**

   Evaluates AI agent performance by running it through predefined scenarios. The task executes the agent with specific inputs and captures its responses for evaluation. Use cases include:

   - Testing chatbot responses
   - Evaluating task completion capabilities
   - Assessing decision-making logic

   Example usage:

   ```bash
   arize-experiment run \
     --name agent-test \
     --dataset agent-scenarios \
     --task execute_agent \
     --evaluator sentiment_classification_accuracy \
     --tag model=gpt-4 \
     --tag temperature=0.7
   ```

### Evaluators

1. **Sentiment Classification Accuracy**

   Evaluates the performance of sentiment classification tasks by:

   - Calculating overall accuracy scores
   - Generating confusion matrices
   - Providing detailed misclassification analysis
   - Identifying systematic errors and biases

   The evaluator compares model predictions against ground truth labels and generates comprehensive performance metrics that are automatically uploaded to your Arize dashboard.

### Output and Results

Results are automatically uploaded to your Arize space, where you can:

- View detailed metrics and evaluations
- Compare results across experiments
- Generate performance reports
- Analyze model behavior

## Project Structure

```
arize_experiment/
├── cli/            # Command-line interface
├── core/           # Core functionality
├── evaluators/     # Evaluation implementations
├── tasks/          # Task implementations
└── tests/          # Test suite
```

## Development

### Setting Up Development Environment

1. Follow the installation steps in the Prerequisites section
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=arize_experiment
```

### Code Quality

Format code using Black:

```bash
black .          # Format files
black . --check  # Check formatting
```

Lint code using Flake8:

```bash
flake8 .
```

### Adding New Components

#### Adding a New Task

1. Create a new file in `arize_experiment/tasks/`
2. Implement the `Task` interface from `core/task.py`
3. Register the task in `cli/commands.py`

#### Adding a New Evaluator

1. Create a new file in `arize_experiment/evaluators/`
2. Implement the evaluation logic
3. Register the evaluator in `cli/commands.py`

## Troubleshooting

### Common Issues

1. **API Authentication Errors**

   - Verify your API keys in `.env`
   - Check your Arize account status
   - Ensure your OpenAI API key has sufficient credits

2. **Python Environment Issues**

   ```bash
   pyenv rehash
   eval "$(pyenv init -)"
   ```

3. **Package Installation Issues**

   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

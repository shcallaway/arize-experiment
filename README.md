# Arize Experiment

A CLI tool for creating and running experiments on Arize.

## Requirements

This tool requires Python 3.10 (Python 3.11 and above are not supported due to dependency constraints). We recommend using `pyenv` to manage Python versions, similar to how `nvm` works for Node.js.

### Installing pyenv

```bash
# Using Homebrew
brew install pyenv

# Add to your shell (add these to ~/.zshrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Reload shell
source ~/.zshrc
```

### Setting up Python 3.10 with pyenv

```bash
# List available Python versions
pyenv install --list | grep " 3.10"

# Install Python 3.10
pyenv install 3.10.13

# Set local Python version (for this project)
pyenv local 3.10.13

# Verify Python version
python --version  # Should show Python 3.10.13
```

### Troubleshooting pyenv

If you encounter issues with pyenv:

1. If `python` command is not found:

```bash
# Rehash pyenv shims
pyenv rehash

# Ensure pyenv is initialized in your shell
eval "$(pyenv init -)"
eval "$(pyenv init --path)"
```

2. If you see "python-build: definition not found":

```bash
# Update pyenv and try again
pyenv update
```

3. For immediate access without shell restart, use the full path:

```bash
~/.pyenv/versions/3.10.13/bin/python
```

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install package
pip install -e .
```

## Authentication

You'll need an Arize API key and space ID to use this tool. Create a `.env` file in the project root and add your API key and space ID:

```bash
ARIZE_API_KEY=
ARIZE_SPACE_ID=
```

You can find your API key and space ID in your Arize account settings.

## Usage

The CLI provides a single command `run` that creates and runs an experiment on Arize:

```bash
# Create and run an experiment
python -m arize_experiment.cli run --name my-experiment --dataset my-dataset

# Get help
python -m arize_experiment.cli --help
python -m arize_experiment.cli run --help
```

### Options

- `--name`, `-n`: Name of the experiment to create (required)
- `--dataset`, `-d`: Name of the dataset to use for the experiment (required)

## Development

1. Clone the repository
2. Install pyenv following the instructions above
3. Install Python 3.10: `pyenv install 3.10.13`
4. Set local Python version: `pyenv local 3.10.13`
5. Create a virtual environment: `python -m venv venv`
6. Activate the virtual environment: `source venv/bin/activate`
7. Install dependencies: `pip install -e .`
8. Create `.env` file with your Arize API key and space ID

### Code Formatting

This project uses Black for code formatting. Black is already included in the development dependencies.

```bash
# Format all Python files in the project
black .

# Check formatting without making changes
black . --check
```

### Switching Python Versions

If you need to switch Python versions:

```bash
# List installed versions
pyenv versions

# Install a different version
pyenv install 3.10.13

# Switch version for this project
pyenv local 3.10.13

# Check Python version
python --version  # Should show Python 3.10.13

# Create new venv with different version
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -e .

# To switch back to the default Python version
pyenv local system
```

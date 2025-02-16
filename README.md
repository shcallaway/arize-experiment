# Arize Experiment

A CLI tool for creating and running experiments on Arize.

## Prerequisites

- Python 3.10 (Python 3.11+ not supported due to dependency constraints)
- pyenv (recommended for Python version management)

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

Create a `.env` file by copying the `.env.example` file.

```bash
cp .env.example .env
```

Fill in the values for the environment variables.

You can find these values in your Arize account settings.

## Usage

Run an experiment:

```bash
arize-experiment run --name my-experiment --dataset my-dataset
```

Options:

- `--name`, `-n`: Experiment name (required)
- `--dataset`, `-d`: Dataset name (required)

## Development

### Code Formatting

Format code using Black:

```bash
black .          # Format files
black . --check  # Check formatting
```

## Code Linting

Lint code using Flake8:

```bash
flake8 .
```

## Troubleshooting

If Python command is not found after installing with pyenv:

```bash
pyenv rehash
eval "$(pyenv init -)"
```

If you see "python-build: definition not found":

```bash
pyenv update
```

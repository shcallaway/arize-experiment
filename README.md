# Arize Experiment

A Python script that creates and runs an experiment on Arize.

## Features

TODO

## Setup Instructions

### 1. Prerequisites

Before starting, ensure you have:

- Python 3.9 or higher installed (tested with Python 3.9.6)
- Arize API key

### 2. Development Setup

The easiest way to set up the development environment is using the provided setup script:

```bash
# Make the setup script executable
chmod +x setup_dev.sh

# Run the setup script
./setup_dev.sh
```

This will:

- Create a Python virtual environment
- Install all dependencies including development tools (pytest, black, flake8)
- Create a .env file from the example template

Alternatively, you can set up manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install dependencies and development tools
python -m pip install -r requirements.txt pytest-asyncio
python -m pip install -e .
```

### 3. Running Tests

The project uses pytest with async support for testing. To run the tests:

```bash
# Activate virtual environment if not already active
source venv/bin/activate

# Run tests with verbose output
python -m pytest tests/ -v

# Run tests with more detailed output
python -m pytest tests/ -vv

# Run a specific test file
python -m pytest tests/test_cache_manager.py -v
```

Note: The tests use mocking extensively, so no API keys or vector store setup is required to run them.

### 5. Environment Configuration

Create a `.env` file in the project root directory with the following settings:

```env
ARIZE_API_KEY=
```

## Running the Script

1. Ensure your virtual environment is activated:

```bash
source venv/bin/activate
```

2. Start the script:

```bash
python -m arize-experiment.cli
```

## Development

### Code Linting

The project uses black for code formatting and flake8 for style checking:

```bash
# Format code with black
black .

# Check code style with flake8
flake8
```

Configuration files:

- `.flake8`: Flake8 configuration with 88 character line length to match black
- `pyproject.toml`: Black configuration and build settings

### Running Tests

The project uses pytest with async support for testing:
The project uses pytest with async support for testing:

```bash
# Run tests with verbose output
python -m pytest tests/ -v

# Run tests with more detailed output
python -m pytest tests/ -vv

# Run a specific test file
python -m pytest tests/test_cache_manager.py -v
```

Note: The tests use mocking extensively, so no API keys or vector store setup is required to run them.

## Exiting

To deactivate the virtual environment when you're done:

```bash
deactivate
```

## Troubleshooting

### Environment Setup Issues

1. If you see "command not found: python", try using `python3` instead
2. For virtual environment issues:
   ```bash
   # If venv exists but seems corrupted
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   ```
3. If you get pip-related errors:
   ```bash
   # Upgrade pip to the latest version
   python -m pip install --upgrade pip
   ```

### Testing Issues

1. If tests fail with missing pytest-asyncio:
   ```bash
   python -m pip install pytest-asyncio
   ```
2. If you get SSL-related warnings with urllib3:
   - This is a known issue with LibreSSL on some systems
   - The warnings can be safely ignored for development
3. For test failures:
   - Run tests with -vv flag for detailed output: `python -m pytest tests/ -vv`
   - Ensure you're using Python 3.9 or higher
   - Make sure all test dependencies are installed

## License

GNU General Public License v3.0 (GPLv3)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

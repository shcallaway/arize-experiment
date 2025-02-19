"""
Tests for the sentiment classification task.
"""

from unittest.mock import Mock, patch

import pytest

from arize_experiment.tasks.sentiment_classification import SentimentClassificationTask


@pytest.fixture
def mock_openai_response() -> Mock:
    """Create a mock OpenAI API response."""
    mock_message = Mock()
    mock_message.content = "positive"

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_completion = Mock()
    mock_completion.choices = [mock_choice]

    return mock_completion


@pytest.fixture
def mock_openai_client(mock_openai_response: Mock) -> Mock:
    """Create a mock OpenAI client."""
    mock_client = Mock()
    mock_chat = Mock()
    mock_completions = Mock()
    mock_completions.create.return_value = mock_openai_response
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat
    return mock_client


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_task_initialization(mock_openai: Mock) -> None:
    """Test that the task initializes with default and custom parameters."""
    # Test with defaults
    task = SentimentClassificationTask()
    assert task.name == "sentiment_classification"
    assert task.model == "gpt-4o-mini"
    assert task.temperature == 0

    # Test with custom parameters
    task = SentimentClassificationTask(
        model="gpt-4", temperature=0.7, api_key="test-key"
    )
    assert task.model == "gpt-4"
    assert task.temperature == 0.7


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_execute_valid_input(mock_openai: Mock, mock_openai_client: Mock) -> None:
    """Test task execution with valid input."""
    mock_openai.return_value = mock_openai_client

    task = SentimentClassificationTask()
    result = task.execute({"input": "This is a great day!"})

    assert result.success
    assert result.output == "positive"
    metadata = result.metadata or {}
    assert metadata.get("model") == "gpt-4o-mini"
    assert metadata.get("temperature") == 0
    assert result.error is None


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_execute_invalid_input_format(
    mock_openai: Mock, mock_openai_client: Mock
) -> None:
    """Test task execution with invalid input format."""
    mock_openai.return_value = mock_openai_client

    task = SentimentClassificationTask()

    # Test with non-dict input
    result = task.execute("not a dict")
    assert not result.success
    assert result.error is not None
    assert "must be a dictionary" in str(result.error).lower()

    # Test with dict missing 'input' key
    result = task.execute({"wrong_key": "text"})
    assert not result.success
    assert result.error is not None
    assert "must be a dictionary with 'input' key" in str(result.error).lower()

    # Test with non-string input
    result = task.execute({"input": 123})
    assert not result.success
    assert result.error is not None
    assert "must be a string" in str(result.error).lower()


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_execute_api_error(mock_openai: Mock) -> None:
    """Test task execution when API call fails."""
    # Create a new mock client that raises an exception
    error_client = Mock()
    error_chat = Mock()
    error_completions = Mock()
    error_completions.create.side_effect = Exception("API Error")
    error_chat.completions = error_completions
    error_client.chat = error_chat

    mock_openai.return_value = error_client

    task = SentimentClassificationTask()
    result = task.execute({"input": "test text"})

    assert not result.success
    assert result.error is not None
    assert "api error" in str(result.error).lower()


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_parse_llm_output(mock_openai: Mock) -> None:
    """Test LLM output parsing."""
    task = SentimentClassificationTask()

    # Test valid outputs
    assert task._parse_llm_output("positive") == "positive"
    assert task._parse_llm_output("NEGATIVE") == "negative"
    assert task._parse_llm_output(" neutral ") == "neutral"

    # Test with whitespace
    assert task._parse_llm_output("  positive\n") == "positive"


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_execute_end_to_end(mock_openai: Mock) -> None:
    """Test complete end-to-end task execution."""
    # Create a fresh mock for this test
    mock_message = Mock()
    mock_message.content = "positive"

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_completion = Mock()
    mock_completion.choices = [mock_choice]

    mock_completions = Mock()
    mock_completions.create.return_value = mock_completion

    mock_chat = Mock()
    mock_chat.completions = mock_completions

    mock_client = Mock()
    mock_client.chat = mock_chat

    mock_openai.return_value = mock_client

    task = SentimentClassificationTask(model="gpt-4", temperature=0.5)

    result = task.execute({"input": "I love this product!"})

    # Verify the API was called with correct parameters
    mock_completions.create.assert_called_once()
    call_args = mock_completions.create.call_args[1]
    assert call_args["model"] == "gpt-4"
    assert call_args["temperature"] == 0.5
    assert len(call_args["messages"]) == 2
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][1]["role"] == "user"
    assert call_args["messages"][1]["content"] == "I love this product!"

    # Verify the result
    assert result.success
    assert result.output == "positive"
    metadata = result.metadata or {}
    assert metadata.get("model") == "gpt-4"
    assert metadata.get("temperature") == 0.5


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_sentiment_classification_init(mock_openai: Mock) -> None:
    """Test sentiment classification task initialization."""
    task = SentimentClassificationTask()
    assert task.model == "gpt-4o-mini"
    assert task.temperature == 0


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_sentiment_classification_execute(mock_openai: Mock) -> None:
    """Test sentiment classification task execution."""
    task = SentimentClassificationTask()
    task._client = Mock()
    task._client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="positive"))]
    )

    result = task.execute({"input": "This is a great day!"})
    assert result.input == {"input": "This is a great day!"}
    assert result.output == "positive"
    assert result.error is None
    metadata = result.metadata or {}
    assert metadata.get("model") == "gpt-4o-mini"
    assert metadata.get("temperature") == 0


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_sentiment_classification_execute_with_error(mock_openai: Mock) -> None:
    """Test sentiment classification task execution with error."""
    task = SentimentClassificationTask()
    task._client = Mock()
    task._client.chat.completions.create.side_effect = Exception("API error")

    result = task.execute({"input": "This is a great day!"})
    assert result.input == {"input": "This is a great day!"}
    assert result.output is None
    assert "API error" in str(result.error)
    metadata = result.metadata or {}
    assert metadata.get("model") == "gpt-4o-mini"
    assert metadata.get("temperature") == 0


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_task_name(mock_openai: Mock) -> None:
    """Test the task name property."""
    task = SentimentClassificationTask()
    assert task.name == "sentiment_classification"


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_task_model(mock_openai: Mock) -> None:
    """Test the model property."""
    task = SentimentClassificationTask(model="gpt-4")
    assert task.model == "gpt-4"


@patch("arize_experiment.tasks.sentiment_classification.OpenAI")
def test_task_temperature(mock_openai: Mock) -> None:
    """Test the temperature property."""
    task = SentimentClassificationTask(temperature=0.7)
    assert task.temperature == 0.7


def test_task_api_key() -> None:
    """Test API key initialization."""
    task = SentimentClassificationTask(api_key="test-key")
    assert task._client is not None

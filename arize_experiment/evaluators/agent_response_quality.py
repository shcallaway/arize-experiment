"""
Evaluator for judging the quality of agent responses using OpenAI's API.
"""

import json
import logging
from typing import Any, List, Optional, Tuple

from arize.experimental.datasets.experiments.types import EvaluationResult
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize_experiment.core.task import TaskResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You will evaluate the quality of an AI agent's response in the context of a "
    "conversation. Consider the following criteria:\n\n"
    "1. Relevance: Does the response directly address the user's query/need?\n"
    "2. Accuracy: Is the information provided correct and well-supported?\n"
    "3. Clarity: Is the response clear, well-structured, and easy to understand?\n"
    "4. Completeness: Does the response fully address all aspects of the query?\n"
    "5. Appropriateness: Is the tone and style appropriate for the context?\n\n"
    "Score the response on a scale of 0 to 1 (use up to 2 decimal places) and provide "
    "a brief explanation of your rating. Format your response as:\n"
    "Score: [0-1]\n"
    "Explanation: [Your explanation]"
)


@EvaluatorRegistry.register("agent_response_quality")
class AgentResponseQualityEvaluator(BaseEvaluator):
    """Evaluates the quality of agent responses using OpenAI's API.

    This evaluator uses GPT models to analyze whether a given agent response
    is appropriate, relevant, and high-quality in the context of the conversation.

    Configuration:
        {
            "type": "agent_response_quality",
            "model": "gpt-4",  # optional
            "temperature": 0.0,  # optional
            "api_key": "sk-..."  # optional
        }
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        api_key: Optional[str] = None,
    ):
        """Initialize the evaluator.

        Args:
            model: OpenAI model to use
            temperature: Model temperature (0-1)
            api_key: Optional OpenAI API key (uses env var if not provided)
        """
        super().__init__()
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def name(self) -> str:
        """Get the evaluator name."""
        return "agent_response_quality"

    def _parse_chat_completion_content(self, content: str) -> Tuple[float, str]:
        """Parse the LLM output to extract the score and explanation.

        Args:
            content: Raw LLM output content

        Returns:
            Tuple of (score: float, explanation: str)

        Raises:
            ValueError: If the output cannot be parsed
        """
        try:
            lines = content.strip().split("\n")
            if len(lines) < 2:
                raise ValueError("Output must contain score and explanation")

            score_line = lines[0].lower()
            if not score_line.startswith("score:"):
                raise ValueError("First line must start with 'Score:'")

            try:
                score = float(score_line.replace("score:", "").strip())
                if not 0 <= score <= 1:
                    raise ValueError("Score must be between 0 and 1")
            except ValueError:
                raise ValueError("Invalid score format")

            explanation = " ".join(lines[1:]).replace("explanation:", "", 1).strip()
            if not explanation:
                raise ValueError("Empty explanation")

            # Remove any remaining "Explanation:" prefix after joining lines
            if explanation.lower().startswith("explanation:"):
                explanation = explanation[len("explanation:") :].strip()

            return score, explanation
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {str(e)}")

    def evaluate(self, task_result: TaskResult) -> EvaluationResult:
        """Evaluate the quality of the agent's response.

        Args:
            task_result: The task result containing conversation context and response

        Returns:
            EvaluationResult containing:
            - score: Quality score between 0 and 1
            - label: String representation of the score
            - explanation: Human-readable explanation of the evaluation
        """
        logger.info("Starting evaluator")
        logger.debug(f"Task result: {task_result}")

        # Extract conversation context and response from task result
        conversation, agent_response = self._parse_task_result(task_result)

        if not conversation:
            raise ValueError("Missing conversation")

        if not agent_response:
            raise ValueError("Missing agent response")

        try:
            # Prepare chat completion messages
            messages: List[ChatCompletionMessageParam] = (
                self._prepare_chat_completion_messages(conversation, agent_response)
            )

            # Create chat completion
            content = self._create_chat_completion(messages)

            # Parse chat completion content into score and explanation
            score, explanation = self._parse_chat_completion_content(content)

            # Convert score to a descriptive label
            label = self._determine_label(score)

            # Return evaluation result
            return EvaluationResult(
                score=score,
                label=label,
                explanation=explanation,
            )
        except Exception as e:
            raise ValueError(f"Agent response quality evaluation failed: {str(e)}")

    def _create_chat_completion(
        self, messages: List[ChatCompletionMessageParam]
    ) -> str:
        """Create a chat completion.

        Args:
            messages: The messages to create the chat completion with

        Returns:
            The content of the chat completion
        """
        logger.debug("Creating chat completion")

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )
        except Exception as e:
            raise ValueError(f"Failed to create chat completion: {str(e)}")

        content = response.choices[0].message.content

        if content is None:
            raise ValueError("LLM returned empty response")

        return content

    def _prepare_chat_completion_messages(
        self, conversation: List[str], agent_response: str
    ) -> List[ChatCompletionMessageParam]:
        """Prepare the messages for the chat completion.

        Args:
            conversation: The conversation to prepare
            agent_response: The agent response to prepare

        Returns:
            The prepared messages
        """
        logger.debug("Preparing chat completion messages")

        formatted_conversation = self._format_conversation_for_chat_completion_messages(
            conversation
        )

        return [
            ChatCompletionSystemMessageParam(
                role="system",
                content=SYSTEM_PROMPT,
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"Conversation:\n{formatted_conversation}\n\n"
                    f"Agent Response to Evaluate:\n{agent_response}"
                ),
            ),
        ]

    def _format_conversation_for_chat_completion_messages(
        self, conversation: List[str]
    ) -> str:
        """Format the conversation for the chat completion messages.

        Args:
            conversation: The conversation to format

        Returns:
            The formatted conversation
        """
        logger.debug(f"Formatting conversation: {conversation}")

        return "\n".join(
            f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
            for i, msg in enumerate(conversation)
        )

    def _parse_task_result(self, task_result: TaskResult) -> Tuple[List[str], str]:
        """Parse the task result to extract the conversation and response.

        Args:
            task_result: The task result to parse

        Returns:
            Tuple of (conversation: List[str], response: str)
        """
        logger.debug(f"Parsing task result: {task_result}")

        return (
            json.loads(task_result.input.get("input", "")),
            task_result.output.get("response", ""),
        )

    def _determine_label(self, score: float) -> str:
        """Determine the label for the given score.

        Args:
            score: The score to determine the label for

        Returns:
            The label for the given score
        """
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "fair"
        elif score >= 0.3:
            return "poor"
        else:
            return "unacceptable"

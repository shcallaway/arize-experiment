"""
Evaluator for judging the quality of agent responses using OpenAI's API.
"""

import json
import logging
from typing import List, Optional, Tuple

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

    def _parse_llm_output(self, text: str) -> Tuple[float, str]:
        """Parse the LLM output to extract the score and explanation.

        Args:
            text: Raw LLM output text

        Returns:
            Tuple of (score: float, explanation: str)

        Raises:
            ValueError: If the output cannot be parsed
        """
        try:
            lines = text.strip().split("\n")
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
        logger.info("Evaluating agent response quality")
        logger.debug(f"Task result: {task_result}")

        # Extract conversation context and response from task result
        conversation, agent_response = self._parse_task_result(task_result)

        if not conversation:
            raise ValueError("Missing conversation")

        if not agent_response:
            raise ValueError("Missing agent response")

        try:
            # Format conversation context
            formatted_conversation = self._format_conversation(conversation)

            messages: List[ChatCompletionMessageParam] = [
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

            response: ChatCompletion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )

            content = response.choices[0].message.content

            if content is None:
                raise ValueError("LLM returned empty response")

            score, explanation = self._parse_llm_output(content)

            # Convert score to a descriptive label
            label = self._determine_label(score)

            return EvaluationResult(
                score=score,
                label=label,
                explanation=explanation,
            )
        except Exception as e:
            error_msg = f"Agent response quality evaluation failed: {str(e)}"
            logger.error(error_msg)
            return EvaluationResult(
                score=0.0,
                label="error",
                explanation=error_msg,
            )

    def _format_conversation(self, conversation: List[str]) -> str:
        """Format the conversation for the LLM.

        Args:
            conversation: The conversation to format

        Returns:
            The formatted conversation
        """
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
        return (
            json.loads(task_result.dataset_row.get("input", "")),
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

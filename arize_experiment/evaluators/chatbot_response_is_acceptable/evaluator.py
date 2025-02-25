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
from pydantic import BaseModel

from arize_experiment.core.evaluator import BaseEvaluator
from arize_experiment.core.evaluator_registry import EvaluatorRegistry
from arize_experiment.core.task import TaskResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You will evaluate the quality of an AI agent's response in the context of a conversation. "
    "Consider the following criteria:\n\n"
    "Relevance (0-1): Does the response directly address the user's query or intent? Is it aligned with the conversation context?\n"
    "1.00: Fully relevant and directly addresses the user's input.\n"
    "0.50: Partially relevant but somewhat off-track.\n"
    "0.00: Completely irrelevant or ignores the user's intent.\n"
    "Accuracy (0-1): Is the information factually correct, consistent with the script/knowledge base, and free from misleading claims?\n"
    "1.00: Completely accurate, well-supported by facts.\n"
    "0.50: Mostly accurate but contains minor inconsistencies.\n"
    "0.00: Incorrect, misleading, or contradictory to known facts.\n"
    "Clarity (0-1): Is the response well-structured, concise, and easy to understand? Does it avoid ambiguity or unnecessary complexity?\n"
    "1.00: Clear, structured, and easily understandable.\n"
    "0.50: Somewhat clear but could be better structured.\n"
    "0.00: Confusing, ambiguous, or difficult to follow.\n"
    "Completeness (0-1): Does the response fully address all aspects of the user's query? Does it omit critical information?\n"
    "1.00: Fully complete, covering all necessary details.\n"
    "0.50: Partially complete, missing some key elements.\n"
    "0.00: Incomplete or unhelpful.\n"
    "Appropriateness (0-1): Is the tone and style suitable for the context? Is it professional, engaging, and not robotic, rude, or unnatural?\n"
    "1.00: Fully appropriate, natural, and well-suited.\n"
    "0.50: Somewhat appropriate but slightly off in tone.\n"
    "0.00: Inappropriate, unnatural, or poorly suited.\n"
    "Score the response for each criterion on a scale of 0 to 1 (use up to 2 decimal places) and provide "
    "a brief explanation of your rating. Your response must be a JSON with the following format:\n"
    "{\n"
    "  \"relevance\": [0-1],\n"
    "  \"accuracy\": [0-1],\n"
    "  \"clarity\": [0-1],\n"
    "  \"completeness\": [0-1],\n"
    "  \"appropriateness\": [0-1],\n"
    "  \"explanation\": \"[Your explanation]\"\n"
    "}"
)


class AcceptabilityEvaluatorFormat(BaseModel):
    relevance: float
    accuracy: float
    clarity: float
    completeness: float
    appropriateness: float
    explanation: str


@EvaluatorRegistry.register("chatbot_response_is_acceptable")
class ChatbotResponseIsAcceptableEvaluator(BaseEvaluator):
    """Evaluates the quality of chatbot responses using OpenAI's API.

    This evaluator uses GPT models to analyze whether a given chatbot response
    is appropriate, relevant, and high-quality in the context of the conversation.

    Configuration:
        {
            "type": "chatbot_response_is_acceptable",
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
        return "chatbot_response_is_acceptable"

    def _parse_llm_output(self, output: dict) -> Tuple[float, str]:
        """Parse the LLM output to extract the scores and explanation.

        Args:
            output (dict): The structured output from the LLM.

        Returns:
            Tuple[float, str]: A tuple containing the average score and explanation

        Raises:
            ValueError: If the output is missing required keys or contains invalid data.
        """
        try:
            # Extract individual scores
            relevance = float(output.get("relevance", 0))
            accuracy = float(output.get("accuracy", 0))
            clarity = float(output.get("clarity", 0))
            completeness = float(output.get("completeness", 0))
            appropriateness = float(output.get("appropriateness", 0))
            explanation = output.get("explanation", "No explanation provided")

            # Calculate average score
            average_score = (relevance + accuracy + clarity + completeness + appropriateness) / 5.0

            return average_score, explanation
        except KeyError as e:
            raise ValueError(f"Failed to parse LLM output: Missing key {str(e)}")

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
            raise ValueError("Missing chatbot response")

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
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "AcceptabilityEvaluatorFormat",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "relevance": {
                                    "description": "Score for relevance (0-1)",
                                    "type": "string",
                                },
                                "accuracy": {
                                    "description": "Score for accuracy (0-1)",
                                    "type": "string",
                                },
                                "clarity": {
                                    "description": "Score for clarity (0-1)",
                                    "type": "string",
                                },
                                "completeness": {
                                    "description": "Score for completeness (0-1)",
                                    "type": "string",
                                },
                                "appropriateness": {
                                    "description": "Score for appropriateness (0-1)",
                                    "type": "string",
                                },
                                "explanation": {
                                    "description": "Explanation of the scores",
                                    "type": "string"
                                }
                            },
                            "additionalProperties": False
                        }
                    }
                }
            )

            content = json.loads(response.choices[0].message.content)

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

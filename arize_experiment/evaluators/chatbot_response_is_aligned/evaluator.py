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
from arize_experiment.evaluators.chatbot_response_is_aligned.prompt import SYSTEM_PROMPT

from pydantic import BaseModel

logger = logging.getLogger(__name__)

class IsAlignedEvaluatorFormat(BaseModel):
    score: bool
    explanation: str


@EvaluatorRegistry.register("chatbot_response_is_aligned")
class ChatbotResponseIsAlignedEvaluator(BaseEvaluator):
    """Evaluates whether a chatbot response is aligned with a conversation script

    This evaluator uses GPT models to analyze whether a given chatbot response
    is aligned with a conversation script.

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
        agent_id: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        api_key: Optional[str] = None,
    ):
        """Initialize the evaluator.

        Args:
            agent_id: Agent ID to evaluate
            model: OpenAI model to use
            temperature: Model temperature (0-1)
            api_key: Optional OpenAI API key (uses env var if not provided)
        """
        if not agent_id:
            raise ValueError("Agent ID must be provided")

        super().__init__()
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._agent_id = agent_id

    @property
    def name(self) -> str:
        """Get the evaluator name."""
        return "chatbot_response_is_aligned"

    def _get_script_mermaid_graph(self, agent_id: str) -> str:
        """Retrieve the Mermaid graph for the specified agent.

        Args:
            agent_id (str): The ID of the agent whose script is to be retrieved.

        Returns:
            str: The Mermaid graph as a string.

        Raises:
            ValueError: If the Mermaid graph file is not found or an error occurs during reading.
        """
        try:
            file_path = f"arize_experiment/evaluators/chatbot_response_is_aligned/mmd/{agent_id}.mmd"
            with open(file_path, "r") as file:
                mermaid_graph = file.read()
            return mermaid_graph
        except FileNotFoundError:
            raise ValueError(f"Mermaid graph file not found for agent_id: {agent_id}")
        except Exception as e:
            raise ValueError(f"An error occurred while reading the mermaid graph file: {str(e)}")

    def _parse_llm_output(self, output: dict) -> Tuple[bool, str]:
        """Extract the score and explanation from the LLM output.

        Args:
            output (dict): The structured output from the LLM.

        Returns:
            Tuple[bool, str]: A tuple containing the score (as a boolean) and the explanation (as a string).

        Raises:
            ValueError: If the output is missing required keys or contains invalid data.
        """
        try:
            score = output.get("score")
            explanation = output.get("explanation")

            if not isinstance(score, bool):
                raise ValueError("Score must be a boolean")

            if not explanation:
                raise ValueError("Empty explanation")

            return score, explanation
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
            mermaid_graph = self._get_script_mermaid_graph(self._agent_id)

            messages: List[ChatCompletionMessageParam] = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=SYSTEM_PROMPT,
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=(
                        f"Mermaid graph of the sales script: \n {mermaid_graph}"
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
                        "name": "IsAlignedEvaluatorFormat",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "score": {
                                    "description": "The score of the response",
                                    "type": "boolean"
                                },
                                "explanation": {
                                    "description": "The explanation of the score",
                                    "type": "string"
                                },
                                "additionalProperties": False,
                            }
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
        """Format the conversation for input to the LLM.

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

    def _determine_label(self, score: bool) -> str:
        """Determine the label for the given score.

        Args:
            score (bool): The score to evaluate.

        Returns:
            str: The label corresponding to the score, either "acceptable" or "unacceptable".
        """
        return "acceptable" if score else "unacceptable"
        
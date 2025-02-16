from typing import Optional
import openai
from arize_experiment.core.task import Task
from arize_experiment.core.task import TaskResult

class SentimentClassificationTask(Task):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        """Initialize the sentiment classification task.
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (optional if set in environment)
        """
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key
        self._validated = False

    @property
    def name(self) -> str:
        """Get the task name."""
        return "sentiment_classification"

    def execute(self, input: str) -> TaskResult:
        """Execute sentiment classification on input text.
        
        Args:
            input: Single text string
            
        Returns:
            TaskResult containing:
            - output: Classification result
            - metadata: Optional processing information
            - error: Any error message if task failed
        """
        try:
            result = self._classify(input)
            return TaskResult(
                output=result,
                metadata={
                        "model": self.model_name,
                    }
            )
        except Exception as e:
            return TaskResult(
                output=None,
                error=f"Sentiment classification failed: {str(e)}"
            )

    def _classify(
        self,
        text: str,
    ) -> str:
        """Internal sentiment classification method.
        
        Args:
            text: Single text string
            
        Returns:
            Classification result
        """
        messages = [
            {
                "role": "system", 
                "content": "You are a sentiment analyzer. Classify the following text as either 'positive', 'negative', or 'neutral'. Respond with just one word."
            },
            {
                "role": "user",
                "content": text
            }
        ]

        response = openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=10,
        )
        sentiment = response.choices[0].message.content.strip().lower()
        # Ensure we only get valid labels
        if sentiment not in ["positive", "negative", "neutral"]:
            sentiment = "neutral"
        
        return sentiment

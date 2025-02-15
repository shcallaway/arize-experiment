from typing import Dict, List, Union, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from arize_experiment.core.task import Task
from arize_experiment.core.task import TaskResult

class SentimentClassificationTask(Task):
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
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

    def validate(self) -> bool:
        """Validate the task configuration.
        
        Checks that:
        1. OpenAI API key is configured
        2. Model is accessible
        
        Returns:
            True if validation succeeds
            
        Raises:
            ValueError: If validation fails
        """
        if self._validated:
            return True

        try:
            # Test API access with minimal request
            openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            self._validated = True
            return True
        except Exception as e:
            raise ValueError(f"OpenAI configuration validation failed: {str(e)}")

    def execute(self, input_data: Union[str, List[str]]) -> TaskResult:
        """Execute sentiment classification on input text(s).
        
        Args:
            input_data: Single text string or list of text strings
            
        Returns:
            TaskResult containing:
            - output: Dictionary with predictions containing input_text and classification
            - metadata: Optional processing information
            - error: Any error message if task failed
        """
        try:
            result = self._predict(input_data)
            # For single text input, return just the first prediction
            if isinstance(input_data, str):
                return TaskResult(
                    output=result["predictions"][0],
                    metadata={
                        "model": self.model_name,
                        "num_texts": 1
                    }
                )
            # For multiple texts, return all predictions
            return TaskResult(
                output=result,
                metadata={
                    "model": self.model_name,
                    "num_texts": len(result["predictions"])
                }
            )
        except Exception as e:
            return TaskResult(
                output=None,
                error=f"Sentiment classification failed: {str(e)}"
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _predict(
        self,
        texts: Union[str, List[str]],
    ) -> Dict[str, List[Dict[str, str]]]:
        """Internal prediction method with retry logic.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Dictionary containing input texts and their predicted sentiments
        """
        # Convert single string to list for batch processing
        if isinstance(texts, str):
            texts = [texts]

        messages = []
        for text in texts:
            messages.append({
                "role": "system",
                "content": "You are a sentiment analyzer. Classify the following text as either 'positive', 'negative', or 'neutral'. Respond with just one word."
            })
            messages.append({
                "role": "user",
                "content": text
            })

        results = []
        for i, msg_pair in enumerate(messages[::2]):  # Process each text's message pair
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[msg_pair, messages[messages.index(msg_pair) + 1]],
                temperature=0,
                max_tokens=10,
            )
            sentiment = response.choices[0].message.content.strip().lower()
            # Ensure we only get valid labels
            if sentiment not in ["positive", "negative", "neutral"]:
                sentiment = "neutral"
            
            results.append({
                "input_text": texts[i],
                "classification": sentiment
            })

        return {
            "predictions": results
        }

    def get_label_mapping(self) -> Dict[str, str]:
        """Get mapping of sentiment labels.
        
        Returns:
            Dictionary of sentiment labels
        """
        return {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral"
        }

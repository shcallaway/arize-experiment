from openai import OpenAI


SENTIMENT_PROMPT = """
You are evaluating whether tone is positive, neutral, or negative

[Message]: {output}

Respond with either "positive", "neutral", or "negative"
"""


def is_positive(output):
    """
    Evaluate whether the given text has a positive sentiment using OpenAI's API.

    Args:
        output (str): The text to evaluate

    Returns:
        float: 1.0 for positive, 0.5 for neutral, 0.0 for negative
    """
    client = OpenAI()

    # Format the prompt with the input text
    prompt = SENTIMENT_PROMPT.format(output=output)

    # Make API call to OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment analysis assistant. "
                "Respond only with 'positive', 'neutral', or 'negative'.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,  # Use deterministic output
    )

    # Get the label from the first choice
    label = response.choices[0].message.content.strip().lower()

    # Convert label to score
    if label == "positive":
        return 1.0
    elif label == "neutral":
        return 0.5
    else:  # negative
        return 0.0

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
        tuple: (score, label, explanation)
            - score (int): 1 for successful evaluation
            - label (str): One of "positive", "neutral", or "negative"
            - explanation (str): The model's explanation for the classification
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

    # Get explanation with a follow-up call
    explanation_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment analysis assistant. "
                "Explain your classification briefly.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label},
            {"role": "user", "content": "Explain why you classified this as " + label},
        ],
        temperature=0,
    )

    explanation = explanation_response.choices[0].message.content.strip()

    return (1, label, explanation)

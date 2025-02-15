import pandas as pd
import numpy as np

def is_number(output):
    """
    Evaluate whether the given input is a numeric value.
    
    Args:
        output (any): The value to evaluate
        
    Returns:
        tuple: (score, label, explanation)
            - score (int): 1 for successful evaluation
            - label (str): "numeric" or "non-numeric"
            - explanation (str): Description of why the value was classified as numeric or non-numeric
    """
    # Convert input to string for consistent handling
    str_output = str(output).strip()
    
    # Handle empty string case
    if not str_output:
        return (
            1,
            "non-numeric",
            f"Input '' cannot be converted to a numeric value"
        )
    
    try:
        # Attempt to convert to numeric using pandas
        result = pd.to_numeric(str_output, errors='raise')
        
        # Determine if it's an integer by checking if it can be converted to int without loss
        is_integer = float(str_output).is_integer()
            
        value_type = "integer" if is_integer else "float"
            
        return (
            1,
            "numeric",
            f"Input '{str_output}' is a valid numeric value ({value_type})"
        )
        
    except (ValueError, TypeError):
        return (
            1,
            "non-numeric",
            f"Input '{str_output}' cannot be converted to a numeric value"
        )

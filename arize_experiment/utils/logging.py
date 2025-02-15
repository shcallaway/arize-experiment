"""
Basic logging setup for arize-experiment.
"""

import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress verbose logging from dependencies
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("arize").setLevel(logging.INFO)

"""
Arize client initialization and management.
"""

from arize.experimental.datasets import ArizeDatasetsClient

from arize_experiment.config import ArizeConfig
from arize_experiment.logging import get_logger

logger = get_logger(__name__)


class ClientError(Exception):
    """Raised when there are issues with the Arize client."""

    pass


def create_client(config: ArizeConfig) -> ArizeDatasetsClient:
    """Create and initialize an Arize datasets client.

    Args:
        config: Arize API configuration

    Returns:
        Configured Arize datasets client instance

    Raises:
        ClientError: If client initialization fails
    """
    try:
        logger.debug("Initializing Arize datasets client")

        # Create client with required configuration
        client = ArizeDatasetsClient(
            api_key=config.api_key,
            developer_key=config.developer_key,
        )

        logger.debug("Successfully initialized Arize datasets client")
        return client

    except Exception as e:
        error_msg = f"Failed to initialize Arize datasets client: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ClientError(error_msg) from e


"""
Arize client initialization and management.
"""

from arize.pandas.logger import Client
import logging

from arize_experiment.config import ArizeConfig

logger = logging.getLogger(__name__)


class ClientError(Exception):
    """Raised when there are issues with the Arize client."""

    pass


def create_client(config: ArizeConfig) -> Client:
    """Create and initialize an Arize client.

    Args:
        config: Arize API configuration

    Returns:
        Configured Arize client instance

    Raises:
        ClientError: If client initialization fails
    """
    try:
        logger.debug(f"Initializing Arize client with space ID: {config.space_id}")

        # Create client with required configuration
        client = Client(
            api_key=config.api_key,
            space_id=config.space_id,
        )

        # Client is initialized with credentials, no need for explicit verification
        logger.debug("Successfully initialized Arize client")
        return client

    except Exception as e:
        error_msg = f"Failed to initialize Arize client: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ClientError(error_msg) from e


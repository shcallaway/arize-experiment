"""
Reusable CLI options for arize-experiment.
"""

import click


def experiment_options(f):
    """Common options for experiment commands."""
    f = click.option(
        "--name",
        "-n",
        required=True,
        help="Name of the experiment to create",
    )(f)

    f = click.option(
        "--dataset",
        "-d",
        required=True,
        help="Name of the dataset to use for the experiment",
    )(f)

    f = click.option(
        "--description",
        help="Optional description of the experiment",
    )(f)

    f = click.option(
        "--tag",
        "-t",
        multiple=True,
        help="Optional tags in key=value format (can be used multiple times)",
    )(f)

    return f


def parse_tags(tag_list):
    """Parse tag options into a dictionary.

    Args:
        tag_list: List of strings in key=value format

    Returns:
        Dict of parsed tags

    Raises:
        click.BadParameter: If tag format is invalid
    """
    if not tag_list:
        return None

    tags = {}
    for tag in tag_list:
        try:
            key, value = tag.split("=", 1)
            tags[key.strip()] = value.strip()
        except ValueError:
            raise click.BadParameter(
                f"Invalid tag format: {tag}. Use key=value format."
            )

    return tags

from setuptools import setup, find_namespace_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="arize-experiment",
    version="0.2.0",  # Hardcoded version number
    author="Sherwood Callaway",
    description="Create and run an experiment on Arize",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["arize_experiment*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "arize-experiment=arize_experiment.cli:main",
        ],
    },
)

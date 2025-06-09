#!/usr/bin/env python3
"""
ActiveCircuitDiscovery: An Active Inference Approach to Circuit Discovery in LLMs
Setup configuration for the research project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('-'):
            requirements.append(line)

setup(
    name="active-circuit-discovery",
    version="1.0.0",
    author="YorK_RP Research Project",
    author_email="research.project@york.ac.uk",
    description="An Active Inference approach to circuit discovery in Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/york-research/active-circuit-discovery",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "full": [
            "sae-lens>=1.0.0",
            "circuit-tracer>=0.1.0", 
            "circuitsvis>=1.0.0"
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950"
        ]
    },
    entry_points={
        "console_scripts": [
            "active-circuit-discovery=experiments.golden_gate_bridge:main",
        ],
    },
    keywords="machine-learning artificial-intelligence mechanistic-interpretability active-inference circuit-discovery transformers",
    project_urls={
        "Bug Reports": "https://github.com/york-research/active-circuit-discovery/issues",
        "Source": "https://github.com/york-research/active-circuit-discovery",
        "Documentation": "https://github.com/york-research/active-circuit-discovery/docs",
    },
)
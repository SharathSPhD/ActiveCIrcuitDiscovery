#!/usr/bin/env python3
"""
ActiveCircuitDiscovery: Enhanced Active Inference Approach to Circuit Discovery in LLMs
Setup configuration for the enhanced research project with statistical validation and prediction systems.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                requirements.append(line)
except FileNotFoundError:
    # Fallback requirements if file doesn't exist
    requirements = [
        "torch>=2.0.0",
        "numpy>=1.21.0", 
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "plotly>=5.0.0",
        "networkx>=2.6.0",
        "PyYAML>=6.0.0",
        "tqdm>=4.62.0",
        "transformers>=4.20.0",
        "transformer-lens>=1.0.0",
        "jaxtyping>=0.2.0",
        "einops>=0.6.0",
        "fancy-einsum>=0.0.1"
    ]

setup(
    name="active-circuit-discovery",
    version="2.0.0",  # Enhanced version
    author="YorK_RP Research Project",
    author_email="research.project@york.ac.uk",
    description="Enhanced Active Inference approach to circuit discovery in Large Language Models with statistical validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/york-research/active-circuit-discovery",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",  # Enhanced to production status
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "enhanced": [
            "statsmodels>=0.13.0",
            "pingouin>=0.5.0",
            "jupyter-widgets>=8.0.0",
            "ipywidgets>=8.0.0",
            "kaleido>=0.2.0"
        ],
        "research": [
            "sae-lens>=1.0.0",
            "circuitsvis>=1.0.0",
            "pymdp>=0.0.1"
        ],
        "full": [
            "statsmodels>=0.13.0",
            "pingouin>=0.5.0",
            "jupyter-widgets>=8.0.0",
            "ipywidgets>=8.0.0", 
            "kaleido>=0.2.0",
            "sae-lens>=1.0.0",
            "circuitsvis>=1.0.0",
            "pymdp>=0.0.1"
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0"
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "notebook>=6.4.0",
            "ipykernel>=6.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "active-circuit-discovery=experiments.golden_gate_bridge:main",
            "acd-experiment=experiments.golden_gate_bridge:main",
            "acd-enhanced=src.experiments.runner:run_golden_gate_experiment",
        ],
    },
    keywords=[
        "machine-learning", "artificial-intelligence", "mechanistic-interpretability", 
        "active-inference", "circuit-discovery", "transformers", "statistical-validation",
        "prediction-systems", "neural-networks", "interpretability", "nlp", "deep-learning",
        "bayesian-inference", "expected-free-energy", "sparse-autoencoders", "research"
    ],
    project_urls={
        "Homepage": "https://github.com/york-research/active-circuit-discovery",
        "Bug Reports": "https://github.com/york-research/active-circuit-discovery/issues",
        "Source": "https://github.com/york-research/active-circuit-discovery",
        "Documentation": "https://github.com/york-research/active-circuit-discovery/blob/main/docs/api_reference.md",
        "Experiments": "https://github.com/york-research/active-circuit-discovery/tree/main/experiments",
        "Google Colab": "https://github.com/york-research/active-circuit-discovery/blob/main/colab_notebook_complete.ipynb",
        "Research Paper": "https://github.com/york-research/active-circuit-discovery/blob/main/YORK_RP_STATUS_REPORT.md"
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "*.yml"],
        "docs": ["*.md"],
        "tests": ["*.py"],
        "": ["*.md", "*.txt", "*.yml", "*.yaml"]
    },
    zip_safe=False,
    # Additional metadata for enhanced version
    platforms=["any"],
    license="MIT",
    license_files=["LICENSE"],
)
"""Setup script for Personal Knowledge Graph Server."""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Personal Knowledge Graph Server - AI-powered knowledge management"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip() 
                for line in f.readlines() 
                if line.strip() and not line.startswith("#")
            ]
    return []

setup(
    name="personal-knowledge-graph-server",
    version="0.1.0",
    description="AI-powered Personal Knowledge Graph Server using OpenRouter API and Neo4j",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Personal Knowledge Graph Team",
    author_email="contact@example.com",
    url="https://github.com/your-username/personal-knowledge-graph-server",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-asyncio>=0.21.1",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.1",
        ],
        "nlp": [
            "spacy>=3.7.2",
            "transformers>=4.34.0",
        ]
    },
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "kg-server=src.mcp_server:main",
            "kg-setup=scripts.setup:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Groupware",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="knowledge-graph ai nlp neo4j openrouter mcp personal-assistant",
    project_urls={
        "Bug Reports": "https://github.com/your-username/personal-knowledge-graph-server/issues",
        "Source": "https://github.com/your-username/personal-knowledge-graph-server",
        "Documentation": "https://github.com/your-username/personal-knowledge-graph-server/wiki",
    },
) 
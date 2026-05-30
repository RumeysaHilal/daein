"""
DAEIN-MFG — Python Paket Kurulum Dosyası
pip install -e .  komutu ile geliştirme modunda kurulur.
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="daein-mfg",
    version="0.5.0",
    author="[Adın SoyAdın]",
    author_email="[email@university.edu]",
    description=(
        "Decentralized Agentic Edge Intelligence Network for "
        "Proactive Fault Prediction in Smart Manufacturing"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/daein-mfg",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "joblib>=1.2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0", "pytest-asyncio>=0.21.0", "pytest-cov>=4.0.0"],
        "viz": ["matplotlib>=3.7.0", "jupyter>=1.0.0"],
        "deploy": ["paho-mqtt>=1.6.0", "ray>=2.5.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "daein-train=scripts.train_models:main",
            "daein-run=scripts.run_simulation:main",
        ],
    },
)

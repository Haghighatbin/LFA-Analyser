"""
Setup configuration for LFA Analyser package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="lfa_analyser",
    version="1.0.0",
    author="Amin Haghighatbin",
    author_email="",
    description="Lateral Flow Assay (LFA) image analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xxxxxxxxxx/lfa-analyser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "streamlit>=1.28.0",
        "Pillow>=10.0.0",
        "plotly>=5.18.0",
        "kaleido>=0.2.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lfa-analyse=lfa_analyser.cli:main",
            "lfa-streamlit=lfa_analyser.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
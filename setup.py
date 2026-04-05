"""
Setup configuration for IFT LIGO Analysis
"""

from setuptools import setup, find_packages

setup(
    name="ift-ligo-analysis",
    version="1.0.0",
    author="Juan Diego Vicente Gabancho",
    author_email="jdvg@physics.org",
    description="Search for f¹ gravitational wave signature in LIGO GWTC-3 data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JuanDiegoVG/IFT-LIGO-Analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7+",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "ift-ligo=ift_ligo:run_complete_analysis",
        ],
    },
)

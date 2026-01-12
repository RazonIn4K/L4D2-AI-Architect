#!/usr/bin/env python3
"""
Setup script for L4D2-AI-Architect

This file is for backward compatibility with pip install.
Configuration is primarily in pyproject.toml.

Usage:
    # Install in development mode
    pip install -e .

    # Install with SDK dependencies
    pip install -e ".[sdk]"

    # Install with full SDK support
    pip install -e ".[sdk-full]"
"""

from setuptools import setup

# All configuration is in pyproject.toml
if __name__ == "__main__":
    setup()

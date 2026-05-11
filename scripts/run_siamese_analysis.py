#!/usr/bin/env python3
"""
Main entry point for Siamese Residual Analysis

This script runs the Siamese neural network analysis by importing and using
the modular components from the siamese_analysis package.
"""

import sys
from siamese_analysis.cli import run_cli

if __name__ == "__main__":
    run_cli() 
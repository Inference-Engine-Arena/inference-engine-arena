#!/usr/bin/env python3
"""
Inference Engine Arena - Main Entry Point

This script provides the main entry point for the Inference Engine Arena CLI.
"""

import sys
from src.cli.commands import main

if __name__ == "__main__":
    sys.exit(main())
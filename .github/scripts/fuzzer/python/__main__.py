#!/usr/bin/env python3
"""Entry point for running as a module: python -m python ..."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

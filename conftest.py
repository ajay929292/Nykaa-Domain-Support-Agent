"""Pytest configuration and pythonpath injection."""

import os
import sys

# Insert project root into sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


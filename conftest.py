"""
conftest.py 

Project-wide fixtures and test utilities.
"""

import sys
from pathlib import Path

# Add the src directory to sys.path for imports in tests
PROJECT_ROOT = Path(__file__).parent

# The src directory is added to sys.path so that test modules can import from it directly
sys.path.insert(0, str(PROJECT_ROOT / "src"))

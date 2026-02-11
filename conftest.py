"""
conftest.py – umístit do kořene projektu (vedle /src a /tests)

Automaticky přidá /src do sys.path, takže testy mohou importovat
moduly jako `from main import ...` bez nutnosti instalace balíčku.
"""

import sys
from pathlib import Path

# Kořen projektu = složka kde leží tento soubor
PROJECT_ROOT = Path(__file__).parent

# Přidej /src na začátek sys.path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

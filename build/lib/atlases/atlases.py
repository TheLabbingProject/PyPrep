from pathlib import Path
from enum import Enum

CURRENT_DIR = Path(__file__).parent.absolute()


class Atlases(Enum):
    megaatlas = CURRENT_DIR / "megaatlas"

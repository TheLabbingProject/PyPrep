from enum import Enum
from pathlib import Path

CURRENT_DIR = Path(__file__)


class Templates(Enum):
    design = CURRENT_DIR.parent.absolute() / "design.fsf"

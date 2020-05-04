from pathlib import Path
from .registrations_functions import invert_warp
from PyPrep.preprocessing import FSLOUTTYPE


class InvertWarp:
    def __init__(self, in_file: Path, ref: Path, out_dir: Path, warp: Path):
        self.in_file = in_file
        self.ref = ref
        self.out_dir = out_dir
        self.warp = warp
        self.out_warp = out_dir / "atlas2highres_warp{FSLOUTTYPE}"
        self.exist = self.out_warp.is_file()

    def invert_warp(self):
        invwarp = invert_warp(self.in_file, self.ref, self.out_warp, self.warp)
        invwarp.run()

    def run(self):
        if not self.exist:
            print("Inverting highres2standard warp...")
            self.invert_warp()
        else:
            print("Inverted highres2standard warp already exists. Continuing.")
        return self.out_warp

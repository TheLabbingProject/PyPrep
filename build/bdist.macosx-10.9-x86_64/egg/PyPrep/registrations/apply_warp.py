from pathlib import Path
from .registrations_functions import *
from PyPrep.preprocessing import FSLOUTTYPE


class ApplyWarp:
    def __init__(self, warp: Path, in_file: Path, ref: Path, out_dir: Path):
        self.warp = warp
        self.in_file = in_file
        self.ref = ref
        self.out_dir = out_dir
        if "highres.nii" in str(in_file):
            atlas_type = "highres"
        elif "Labels.nii" in str(in_file):
            atlas_type = "labels"
        self.out_file = out_dir / f"{atlas_type}_atlas2highres{FSLOUTTYPE}"
        self.exist = self.out_file.is_file()

    def apply_warp(self):
        aw = apply_warp(self.in_file, self.ref, self.warp, self.out_file)
        aw.run()

    def run(self):
        if not self.exist:
            print("Apply non-linear warp on atlas file")
            self.apply_warp()
        else:
            print("Warped atlas file already exists. Continuing.")
        return self.out_file

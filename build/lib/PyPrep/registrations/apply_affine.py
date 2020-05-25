from pathlib import Path
from .registrations_functions import apply_affine
from PyPrep.preprocessing import FSLOUTTYPE


class ApplyAffine:
    def __init__(self, dwi: Path, labels: Path, aff: Path, out_dir: Path):
        self.dwi = dwi
        self.labels = labels
        self.aff = aff
        self.out_dir = out_dir
        self.out_file = out_dir / f"labels_atlas2dwi{FSLOUTTYPE}"
        self.exist = self.out_file.is_file()

    def apply_affine(self):
        ax = apply_affine(self.dwi, self.labels, self.aff, self.out_file)
        ax.run()

    def run(self):
        if not self.exist:
            print("Applying affine transformation on labels image...")
            self.apply_affine()
        else:
            print("Labels image already exist. Continuing.")

from pathlib import Path
from .registrations_functions import highres2dwi
from PyPrep.preprocessing import FSLOUTTYPE


class Resample2DWI:
    def __init__(
        self, epi: Path, standard_in_highres: Path, out_dir: Path, epi_type: str
    ):
        self.epi = epi
        self.standard_in_highres = standard_in_highres
        self.out_dir = out_dir
        self.epi_type = epi_type
        if "highres_atlas" in str(standard_in_highres):
            atlas_type = "highres"
        elif "labels_atlas" in str(standard_in_highres):
            atlas_type = "labels"
        self.out_file = out_dir / f"{atlas_type}_atlas2{epi_type}{FSLOUTTYPE}"
        self.exist = self.out_file.is_file()
        self.out_matrix_file = out_dir / f"{atlas_type}_atlas2{epi_type}_affine.mat"

    def resample_2_dwi(self):
        flt = highres2dwi(
            self.standard_in_highres, self.epi, self.out_file, self.out_matrix_file
        )
        flt.run()

    def run(self):
        if not self.exist:
            print(f"Resampling atlas from highres space to {self.epi_type} image...")
            self.resample_2_dwi()
        else:
            print(f"Atlas in {self.epi_type} space already exists. Continuing.")
        return self.out_file

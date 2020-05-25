from pathlib import Path
from logs import messages
from PyPrep.utils import check_files_existence, FSLOUTTYPE
import os


class StructuralPreprocessing:
    def __init__(self, derivatives: Path, subj: str, anat: Path):
        self.init_anat = Path(derivatives / subj / "anat")
        self.anat = anat
        self.anat_dir = Path(derivatives / subj / "anat" / "prep")
        self.highres_brain = (
            Path(f"{self.anat_dir}.anat") / f"T1_biascorr_brain{FSLOUTTYPE}"
        )
        self.highres_mask = (
            Path(f"{self.anat_dir}.anat") / f"T1_biascorr_brain_mask{FSLOUTTYPE}"
        )
        self.highres = Path(f"{self.anat_dir}.anat") / f"T1_biascorr{FSLOUTTYPE}"
        self.exist = check_files_existence(
            [self.highres, self.highres_mask, self.highres_brain]
        )

    def __str__(self):
        str_to_print = messages.PREPROCESSANAT.format(
            init_anat=self.init_anat, anat=self.anat, anat_dir=self.anat_dir
        )
        return str_to_print

    def preprocess_anat(self):
        cmd = f"fsl_anat -i {self.anat} -o {self.anat_dir}"
        print(cmd)
        os.system(cmd)

    def run(self):
        if not self.exist:
            print("Performing structural preprocessing using fsl_anat:")
            self.preprocess_anat()
        else:
            print("Structural preprocessing already done. Continuing.")
        return self.highres, self.highres_brain, self.highres_mask

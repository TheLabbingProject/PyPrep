from pathlib import Path
from PyPrep.utils import *
from PyPrep.utils.file_utils import check_files_existence
from PyPrep.functional import fmri_prep_functions as fmri_methods
from logs import messages
import os


class FEAT:
    def __init__(
        self,
        subj: str,
        epi_file: Path,
        highres: Path,
        fieldmap_brain: Path,
        fieldmap_rad: Path,
        temp_design: Path,
        derivatives_dir: Path,
    ):
        self.subj = subj
        self.design_template = temp_design
        self.epi_file = epi_file
        self.highres = highres
        self.fieldmap_brain = fieldmap_brain
        self.fieldmap_rad = fieldmap_rad
        self.subj_design = derivatives_dir / subj / "scripts" / "func_design.fsf"
        self.out_feat = derivatives_dir / subj / "func" / f"{epi_file.name}.feat"
        self.exist = check_files_existence([self.subj_design])

    def __str__(self):
        str_to_print = messages.FEAT.format(
            fmri=self.epi_file.name,
            highres=self.highres.name,
            fieldmap_brain=self.fieldmap_brain.name,
            fieldmap_rad=self.fieldmap_rad.name,
            subj_design=self.subj_design.name,
            out_feat=self.out_feat,
        )
        return str_to_print

    def generate_fsf(self):
        gen_fsf = fmri_methods.FeatDesign(
            out_dir=self.out_feat,
            in_file=self.epi_file,
            highres_brain=self.highres,
            temp_design=self.design_template,
            out_design=self.subj_design,
            fieldmap_rad=self.fieldmap_rad,
            fieldmap_brain=self.fieldmap_brain,
        )
        fsf = gen_fsf.run()

    def run(self):
        if not self.exist:
            print("Generating .fsf FEAT's design file.")
            self.generate_fsf()
        else:
            print(".fsf FEAT's design file already exists. Continuing.")
        if not self.out_feat.is_dir():
            print("Performing FSL's FEAT analysis.")
            os.system(f"feat {self.subj_design}")
        else:
            print("FEAT analysis already done. Continuing.")

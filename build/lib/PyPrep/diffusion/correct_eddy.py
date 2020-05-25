from pathlib import Path
from PyPrep.utils import check_files_existence, FSLOUTTYPE
from logs import messages
from PyPrep.diffusion import dmri_prep_functions as dmri_methods
import os


class PreprocessDWI:
    def __init__(self, degibbs: Path, phasediff: Path):
        """
        Initiate mrtrix3's dwipreproc script for eddy currents and various other geometric corrections
        Arguments:
            degibbs {Path} -- [Gibbs rings removed DWI image]
            phasediff {Path} -- [Dual encoded (AP-PA) DWI image]
        """
        self.degibbs = degibbs
        self.phasediff = phasediff
        self.out_file = Path(degibbs.parent / "dwi_preprocessed.mif")
        self.exist = check_files_existence([self.out_file])

    def __str__(self):
        str_to_print = messages.DWIPREPROCESS.format(
            degibbs=self.degibbs, phasediff=self.phasediff, preprocessed=self.out_file
        )

    def preprocess(self):
        cmd = dmri_methods.dwi_prep(self.degibbs, self.phasediff, self.out_file)
        os.system(cmd)

    def run(self):
        if not self.exist:
            print(
                "Performing various geometric corrections of DWIs using Mrtrix3's dwipreproc script"
            )
            self.preprocess()
        else:
            print(
                "Already used Mrtrix3's dwipreproc script to perform geometric corrections of DWI image."
            )
        return self.out_file

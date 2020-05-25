from pathlib import Path
from .utillities import FSLOUTTYPE
from .file_utils import check_files_existence
from PyPrep.functional import fmri_prep_functions as fmri_methods


class MotionCorrection:
    """
        Perform motion correction with FSL's MCFLIRT
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]
            in_file {Path} -- [Path to a 4D file to perform motion correction (realignment) on.]
            out_file {Path} -- [Path to output 4D image.]
        """

    def __init__(self, in_file: Path, out_file: Path):
        self.in_file = in_file
        self.out_file = out_file
        self.exist = check_files_existence([out_file])

    def motion_correct(self):
        mot_cor = fmri_methods.motion_correct(self.in_file, self.out_file)
        mot_cor.run()

    def run(self):
        if not self.exist:
            print("Performing motion correction using FSL`s MCFLIRT")
            self.motion_correct()
        else:
            print("Motion correction already done.")
        return self.out_file

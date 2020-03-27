import glob, os, shutil
from PyPrep import (
    Mrtrix3_methods as MRT_Methods,
    Preproc_Methods as Methods,
)
from BidsConverter.Code import ListFiles
import subprocess
from bids_validator import BIDSValidator
import nibabel as nib

MOTHER_DIR = "/Users/dumbledore/Desktop/bids_dataset"

class BidsPrep:
    def __init__(
        self, mother_dir: str, seq: str, subj: str = None, out_dir: str = None
    ):
        """
        Initiate the BIDS preprocessing class with either a specific subject or None = all subjects (default)
        Arguments:
            mother_dir {str} -- [Path to a BIDS compliant directory (should contain "mother_dir/sub-xx")]
            seq (str) -- ["func","dwi","anat". Specify the modality of preprocessing you would like to produce.]
        Keyword Arguments:
            subj {str} -- ["sub-xx" for specific subject or None for all subjects] (default: {None})

        """
        self.mother_dir = mother_dir
        self.subjects = list()
        self.seq = seq
        if subj:
            self.subjects.append(subj)
        else:
            self.subjects = [
                cur_subj.split(os.sep)[-1]
                for cur_subj in glob.glob(f"{self.mother_dir}/sub-*")
            ]
        if not out_dir:
            out_dir = f"{os.path.dirname(mother_dir)}/derivatives"
        self.out_dir = out_dir

    def check_bids(self, mother_dir=None):
        if not mother_dir:
            mother_dir = self.mother_dir
        """
        Validates BidsPrep.mother_dir as a BIDS compatible directory.

        Keyword Arguments:
            mother_dir {[type]} -- [Path to a BIDS compliant directory.] (default: {self.mother_dir})
        """
        print(f"Validating {mother_dir} as a BIDS compatible directory...")
        try:
            validator = subprocess.check_output(
                ["bids-validator", "--ignoreWarnings", f"{mother_dir}"]
            )
        except:
            validator = "Incompatible BIDS directory"
        if not "BIDS compatible" in str(validator):
            raise ValueError(
                f"This path is not a BIDS comaptible direcory.\n\t Please make sure it follows the BIDS specifications before going through preprocessing procedures."
            )
        else:
            print("This is a BIDS compliant directory! Moving on.")

    def set_output_dir(self):
        """
        Manipulate the output directory to resemble the inpur, BIDS compatible directory.
        """
        in_name = self.mother_dir.split(os.sep)[-1]
        out_name = self.out_dir.split(os.sep)[-1]
        dirs = glob.glob(f"{self.mother_dir}/sub*/*")
        for d in dirs:
            new_d = d.replace(in_name, out_name)
            if not os.path.isdir(new_d):
                os.makedirs(new_d)
        print(f"Sorted output directory to resemble input, BIDS compatible one:")
        ListFiles.list_files(self.out_dir)

    def BrainExtract(self, subj: str, in_file: str, seq: str = "anat"):
        """
        Perform brain extraction using FSL's bet.
        *** Need improvement - maybe use cat12
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]
            in_file {str} -- [Path to file to perform brain extraction on.]
        Keyword Arguments:
            subdir {str} -- ['anat','dwi' or 'func' perform brain extraction for an image in specific subdirectory in mother_dir.] (default: {'anat'})
        """
        f_name = in_file.split(os.sep)[-1]
        new_f = os.path.splitext(os.path.splitext(f_name)[0])[0]
        out_file = f"{self.out_dir}/{subj}/{seq}/{new_f}_brain.nii"
        print("Performing brain extractoin using FSL`s bet")
        print(f"Input file: {in_file}")
        print(f"Output file: {out_file}")
        bet = Methods.run_BET(in_file, out_file)
        bet = bet.run()
        return bet

    def MotionCorrect(self, subj: str, in_file: str, seq: str = "func"):
        """
        Perform motion correction with FSL's MCFLIRT
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]
            in_file {str} -- [Path to a 4D file to perform motion correction (realignment) on.]

        Keyword Arguments:
            seq {str} -- ['func' or 'dwi', Modality to perform motion correction on.] (default: {"func"})
        """
        img = nib.load(in_file)
        is_4d = img.ndim > 3
        if is_4d:
            f_name = in_file.split(os.sep)[-1]
            new_f = os.path.splitext(os.path.splitext(f_name)[0])[0]
            out_file = f"{self.out_dir}/{subj}/{seq}/{new_f}_MotionCorrected.nii"
            mot_cor = Methods.MotionCorrection(in_file=in_file, out_file=out_file)
            print("Performing motion correction using FSL`s MCFLIRT")
            print(f"Input file: {in_file}")
            print(f"Output file: {out_file}")
            mcflt = mot_cor.run()
        return mcflt


    def run(self):
        print("Iinitiating preprocessing procedures...")
        print(f"Input directory: {self.mother_dir}")
        self.check_bids(self.mother_dir)
        self.set_output_dir()
        for subj in self.subjects:
            (
                anat,
                func,
                sbref,
                dwi,
                bvec,
                bval,
                phasediff,
            ) = Methods.load_initial_files(self.mother_dir, subj)
            self.BrainExtract(subj, anat)
            if func:
                self.MotionCorrect(subj, func, "func")
            if dwi:
                self.MotionCorrect(subj, dwi, "dwi")


# to run ICA-AROMA:
# 1. bet T1
# 2. epi-reg from rest to t1 (.mat)
# 3. fnirt from t1 to mni (.nii.gz)
# 4. mcflirt rest (.par)

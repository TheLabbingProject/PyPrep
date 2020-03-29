import glob, os, shutil
from PyPrep.Code import (
    Mrtrix3_methods as MRT_Methods,
    Preproc_Methods as Methods,
)
from BidsConverter.Code import ListFiles
import subprocess
from bids_validator import BIDSValidator
import nibabel as nib
import tkinter as tk
from tkinter import filedialog
import time

MOTHER_DIR = "/Users/dumbledore/Desktop/bids_dataset"
ATLAS = "/Users/dumbeldore/Desktop/megaatlas"
DESIGN = os.path.abspath("./PyPrep/templates/design.fsf")


class BidsPrep:
    def __init__(
        self,
        mother_dir: str,
        seq: str,
        subj: str = None,
        out_dir: str = None,
        atlas: str = None,
        design: str = DESIGN,
    ):
        """
        Initiate the BIDS preprocessing class with either a specific subject or None = all subjects (default)
        Arguments:
            mother_dir {str} -- [Path to a BIDS compliant directory (should contain "mother_dir/sub-xx")]
            seq (str) -- ["func","dwi","anat". Specify the modality of preprocessing you would like to produce.]
        Keyword Arguments:
            subj {str} -- ["sub-xx" for specific subject or None for all subjects] (default: {None})
            atlas {str} -- [path to atlas` directory. should contain "Highres" and "Labels" files.]
        """
        self.mother_dir = mother_dir
        self.subjects = list()
        self.seq = seq
        self.design = design
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
        self.atlas = atlas
        self.subjects.sort()
        if not atlas:
            self.highres_atlas, self.labels_atlas = self.get_atlas()
        else:
            for f in os.listdir(atlas):
                if "highres.nii" in f:
                    self.highres_atlas = f"{atlas}/{f}"
                elif "Labels.nii" in f:
                    self.labels_atlas = f"{atlas}/{f}"

    def get_atlas(self):
        """
        GUI-based request from user to guide to atlas directory (megaatlas)
        """
        if not self.atlas:
            root = tk.Tk()
            root.withdraw()
            atlas = filedialog.askdirectory(
                title="Please choose the folder containing your megaatlas files."
            )
        for f in os.listdir(atlas):
            if "highres.nii" in f:
                highres_atlas = f"{atlas}/{f}"
            elif "Labels.nii" in f:
                labels_atlas = f"{atlas}/{f}"
        return highres_atlas, labels_atlas

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

    def set_output_dir(self, subj: str):
        """
        Manipulate the output directory to resemble the inpur, BIDS compatible directory.
        """
        in_name = self.mother_dir.split(os.sep)[-1]
        out_name = self.out_dir.split(os.sep)[-1]
        dirs = glob.glob(f"{self.mother_dir}/{subj}/*")
        dirs.append(f"{os.path.dirname(dirs[0])}/atlases")
        dirs.append(f"{os.path.dirname(dirs[0])}/scripts")
        for d in dirs:
            new_d = d.replace(in_name, out_name)
            if not os.path.isdir(new_d):
                os.makedirs(new_d)
        print(f"Sorted output directory to resemble input, BIDS compatible one:")
        ListFiles.list_files(self.out_dir)

    def check_file(self, in_file):
        if os.path.isfile(in_file):
            return False
        else:
            return True

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
        if seq == "anat":
            shutil.copy(in_file, f"{self.out_dir}/{subj}/{seq}")
        out_file = f"{self.out_dir}/{subj}/{seq}/{new_f}_brain.nii"
        if self.check_file(out_file):
            print("Performing brain extractoin using FSL`s bet")
            print(f"Input file: {in_file}")
            print(f"Output file: {out_file}")
            bet = Methods.run_BET(in_file, out_file)
            bet = bet.run()
        return out_file

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
            if self.check_file(out_file):
                mot_cor = Methods.MotionCorrection(in_file=in_file, out_file=out_file)
                print("Performing motion correction using FSL`s MCFLIRT")
                print(f"Input file: {in_file}")
                print(f"Output file: {out_file}")
                mcflt = mot_cor.run()
                # return mcflt

    def prep_feat(self, subj: str, epi_file: str, highres: str):
        """
        Perform FSL's feat preprocessing
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]
            epi_file {str} -- [Path to a 4D file use as input for FSL's feat.]
            highres {str} -- [Path to brain-extracted ('_brain') highres structural file]

        Returns:
            [str] -- [Path to .feat directory]
        """
        temp_design = self.design
        if "dwi" in epi_file:
            subj_design = f"{self.out_dir}/{subj}/scripts/dwi_design.fsf"
            out_feat = f"{self.out_dir}/{subj}/dwi/dwi.feat"
        elif "func" in epi_file:
            subj_design = f"{self.out_dir}/{subj}/scripts/func_design.fsf"
            out_feat = f"{self.out_dir}/{subj}/func/func.feat"
        GenFsf = Methods.feat_design(
            out_feat, epi_file, highres, temp_design, subj_design
        )
        fsf = GenFsf.run()
        if not os.path.isdir(out_feat):
            print(f"Running FEAT for {subj}")
            print(f"input: {epi_file.split(os.sep)[-1]}")
            os.system(f"feat {fsf}")
        return out_feat

    def coregister(self, subj: str, in_file: str, ref: str, proc: int):
        """
        Coregistration between 4D image to structural
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]
            in_file {str} -- [file to linear registrate to {ref}]
            ref {str} -- [path to reference file]
            proc {int} -- [0 for highres2lowres, 1 for atlasHighres2highres, 2 for atlasLabels2highres]
        """
        if proc == 0:
            out_dir = f"{self.out_dir}/{subj}/anat"
        else:
            out_dir = f"{self.out_dir}/{subj}/atlases"
        flt = Methods.FLIRT(in_file, ref, out_dir, proc)
        flt.run()

    def gen_subj_atlas(self, subj: str, highres: str, epi: str):
        """
        Generate transformed atlas in subject's space.
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]
            highres {str} -- [Path to subject's structural, high-resolution image]
            epi {str} -- [Path to subject's epi, low-resolution 4D image]
        """
        for in_file, ref, proc in zip(
            [highres, self.highres_atlas, self.labels_atlas],
            [epi, highres, highres],
            [0, 1, 2],
        ):
            self.coregister(subj, in_file, ref, proc)
        print("Generated subject-space atlas:\n")
        ListFiles.list_files(f"{self.out_dir}/{subj}")

    def load_transforms(self, feat: str):
        reg_dir = f"{feat}/reg"
        for f in os.listdir(reg_dir):
            if "highres2example_func.mat" in f:
                highres2func_aff = f"{reg_dir}/{f}"
            elif "standard2highres.mat" in f:
                standard2highres_aff = f"{reg_dir}/{f}"

    def run(self):
        """
        Complete preprocessing procedure
        """
        print("Iinitiating preprocessing procedures...")
        print(f"Input directory: {self.mother_dir}")
        print(f"Output directory: {self.out_dir}")
        self.check_bids(self.mother_dir)

        for subj in self.subjects:
            t = time.time()
            self.set_output_dir(subj)
            (
                anat,
                func,
                sbref,
                dwi,
                bvec,
                bval,
                phasediff,
            ) = Methods.load_initial_files(self.mother_dir, subj)
            highres_brain = self.BrainExtract(subj, anat)
            if func:
                self.MotionCorrect(subj, func, "func")
                feat = self.prep_feat(subj, func, highres_brain)
            if dwi:
                self.MotionCorrect(subj, dwi, "dwi")
                feat = self.prep_feat(subj, dwi, highres_brain)
            # self.gen_subj_atlas(subj, highres_brain, dwi)

            elapsed = (time.time() - t) / 60
            print("%s`s preproceesing procedures took %.2f minutes." % (subj, elapsed))


# to run ICA-AROMA:
# 1. bet T1
# 2. epi-reg from rest to t1 (.mat)
# 3. fnirt from t1 to mni (.nii.gz)
# 4. mcflirt rest (.par)

import glob
import nibabel as nib
import os
import shutil
import subprocess
import time
import tkinter as tk
from bids_validator import BIDSValidator
from pathlib import Path
from PyPrep.code import dmri_prep_functions as dmri_methods
from PyPrep.code import fmri_prep_functions as fmri_methods
from PyPrep.atlases.atlases import Atlases
from PyPrep.templates.templates import Templates
from tkinter import filedialog
from PyPrep.logs import messages
import logging

ATLAS = Atlases.megaatlas.value
DESIGN = Templates.design.value
FSLOUTTYPE = ".nii.gz"


def check_files_existence(files: list):
    """
    Checks the existence of a list of files (all should exist)
    Arguments:
        file {Path} -- [description]

    Returns:
        [type] -- [description]
    """
    exist = all(Path(f).exists() for f in files)
    return exist


class CheckBids:
    """
    Validates {bids_dir} as a BIDS compatible directory or raises an error otherwise.

    Keyword Arguments:
        bids_dir {[Path]} -- [Path to a BIDS compliant directory.] (default: {self.mother_dir})
    """

    def __init__(self, bids_dir: Path):
        self.bids_dir = bids_dir

    def __str__(self):
        str_to_print = messages.BIDSVALIDATOR.format(bids_dir=self.bids_dir)
        return str_to_print

    def validate_bids(self):
        try:
            validator = subprocess.check_output(
                ["bids-validator", "--ignoreWarnings", f"{self.bids_dir}"]
            )
        except:
            validator = "Incompatible BIDS directory"
        return validator

    def run(self):
        print(f"Validating {self.bids_dir} as a BIDS compatible directory...")
        validator = self.validate_bids()
        if not "BIDS compatible" in str(validator):
            raise ValueError(
                f"This path is not a BIDS comaptible direcory.\n\t Please make sure it follows the BIDS specifications before going through preprocessing procedures."
            )
        else:
            print("This is a BIDS compliant directory! Moving on.")


class DisplayablePath(object):
    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)),
            key=lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path, parent=displayable_root, is_last=is_last, criteria=criteria
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ["{!s} {!s}".format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return "".join(reversed(parts))


class GenerateOutputDirectory:
    """Generates a derivative'output directory at {derivatives_dir} that resesmbles {bids_dir} tree."""

    def __init__(self, bids_dir: Path, derivatives_dir: Path, subj: str):
        self.bids_dir = bids_dir
        self.derivatives_dir = derivatives_dir
        self.subj = subj

    def generate_directories(self):
        in_name = self.bids_dir.name
        out_name = self.derivatives_dir.name
        dirs = glob.glob(f"{Path(self.bids_dir / self.subj)}/*")
        dirs.append(f"{os.path.dirname(dirs[0])}/atlases")
        dirs.append(f"{os.path.dirname(dirs[0])}/scripts")
        for d in dirs:
            new_d = d.replace(in_name, out_name)
            if not os.path.isdir(new_d):
                os.makedirs(new_d)

    def __str__(self):
        str_to_print = messages.GENERATEOUTPUT.format(
            bids_dir=self.bids_dir, derivatives_dir=self.derivatives_dir
        )
        return str_to_print

    def run(self):
        self.generate_directories()
        print(f"Output directory`s tree:")
        paths = DisplayablePath.make_tree(self.derivatives_dir)
        for path in paths:
            print(path.displayable())


class GenerateFieldMap:
    """
    Generate Field Maps using dwi's AP and phasediff's PA scans
    Arguments:
        AP {Path} -- [path to dwi's AP file]
        PA {Path} -- [path to PA phasediff file]
        outdir {Path} -- [path to output directory (outdir/sub-xx/fmap)]
    """

    def __init__(self, subj: str, AP: Path, PA: Path, derivatives_dir: Path):
        self.subj = subj
        self.AP = AP
        self.PA = PA
        self.out_dir = derivatives_dir / subj / "fmap"
        self.initiate_output_files()

    def initiate_output_files(self):
        self.merged = self.out_dir / f"merged_phasediff{FSLOUTTYPE}"
        self.datain = self.out_dir / "datain.txt"
        self.index_file = self.out_dir / "index.txt"
        self.fieldmap = self.out_dir / f"fieldmap{FSLOUTTYPE}"
        self.fieldmap_rad = Path(
            self.fieldmap.parent / f"{Path(self.fieldmap.stem).stem}_rad{FSLOUTTYPE}"
        )
        self.fieldmap_mag = Path(
            self.fieldmap.parent
            / f"{Path(self.fieldmap.stem).stem}_magnitude{FSLOUTTYPE}"
        )
        self.fieldmap_mag_brain = Path(
            self.fieldmap.parent
            / f"{Path(self.fieldmap.stem).stem}_magnitude_brain{FSLOUTTYPE}"
        )

    def __str__(self):
        str_to_print = messages.GENERATEFIELDMAP.format(
            output_dir=self.out_dir,
            AP=self.AP.name,
            PA=self.PA.name,
            datain=self.datain.name,
            index=self.index_file.name,
            fieldmap=self.fieldmap.name,
            fieldmap_rad=self.fieldmap_rad.name,
            fieldmap_mag=self.fieldmap_mag.name,
            fieldmap_brain=self.fieldmap_mag_brain.name,
        )
        return str_to_print

    def merge_phasediff(self):
        """Combine two images into one 4D file
        """
        exist = check_files_existence([self.merged])
        if not exist:
            print("generating dual-phase encoded image.")
            merger = fmri_methods.merge_phases(self.AP, self.PA, self.merged)
            merger.run()
        else:
            print("Dual-phase encoded image already exists. Continuing.")

    def generate_datain(self):
        """Generate datain.txt file for topup
        """
        exist = check_files_existence([self.datain])
        if not exist:
            print("Generating datain.txt with dual-phase data.")
            fmri_methods.generate_datain(self.AP, self.PA, self.datain)
        else:
            print("datain.txt already exists. Continuing.")

    def generate_index(self):
        """Generates index.txt file, needed to run eddy-currents corrections.
        """
        exist = check_files_existence([self.index_file])
        if not exist:
            print("Generating index.txt file, for later eddy-currents correction.")
            fmri_methods.generate_index(self.AP, self.index_file)
        else:
            print("index.txt file already exists. Continuing.")

    def perform_top_up(self):
        """Generate Fieldmap
        """
        exist = check_files_existence([self.fieldmap_mag])
        if not exist:
            print("Using FSL's TopUp to generate fieldmap images.")
            fmri_methods.top_up(self.merged, self.datain, self.fieldmap)
        else:
            print("Fieldmap images already exists. Continuing.")

    def brain_extract(self):
        """Extract fieldmap_mag's brain"""
        exist = check_files_existence([self.fieldmap_mag_brain])
        if not exist:
            print(
                "Using FSL's BET to generate brain-extracted fieldmap magnitude image."
            )
            bet = BrainExtraction(
                in_file=self.fieldmap_mag,
                out_file=self.fieldmap_mag_brain,
                seq=self.seq,
            )
            fieldmap_brain = bet.run()

    def run(self):
        for procedure in [
            self.merge_phasediff,
            self.generate_datain,
            self.generate_index,
            self.perform_top_up,
            self.brain_extract,
        ]:
            procedure()
        return self.fieldmap_rad, self.fieldmap_mag_brain, self.index_file, self.datain


class BrainExtraction:
    """
    Perform brain extraction using FSL's BET.
    Arguments:
        in_file {Path} -- [Path to input nifti image]
    Keyword Arguments:
        out_file {Path} -- [Path to output nifti image] (default: {None})

    """

    def __init__(self, in_file: Path, seq: str, out_file: Path = None):
        self.in_file = in_file
        self.out_file = out_file
        self.seq = seq
        if not out_file:
            self.exist = False
        else:
            self.exist = check_files_existence([out_file])

    def __str__(self):
        str_to_print = messages.BETBRAINEXTRACTION.format(
            in_file=self.in_file, out_file=self.out_file
        )

    def brain_extract(self):
        bet, self.out_file = fmri_methods.bet_brain_extract(self.in_file, self.out_file)
        bet.run()

    def run(self):
        if not self.exist:
            print("Performing brain extractoin using FSL`s bet")
            self.brain_extract()
        else:
            print("Brain extraction already done.")
            return self.out_file


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
        self.out_feat = derivatives_dir / subj / "func" / "func.feat"
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
            temp_design=self.temp_design,
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


class InitiateMrtrix:
    """
    Convert niftis to Mrtrix`s .mif files
    Arguments:
        mrt_folder {Path} -- [Path to subjects' Mrtrix preprocessing directory]
        dwi {Path} -- [Path to motion corrected dwi file]
        mask {Path} -- [Path to field magnitude`s brain mask]
        anat {Path} -- [Path to subject`s strctural image]
        bvec {Path} -- [Path to dwi`s .bvec file]
        bval {Path} -- [Path to dwi`s .bval file]
        phasediff {Path} -- [Path to opposite-phased dwi image]
    """

    def __init__(
        self,
        subj: str,
        derivatives: Path,
        dwi: Path,
        mask: Path,
        anat: Path,
        bvec: Path,
        bval: Path,
        phasediff: Path,
    ):
        self.subj = subj
        self.mrtrix_dir = Path(derivatives / subj / "dwi" / "Mrtrix_prep")
        self.dir_exists = self.mrtrix_dir.is_dir()
        self.dwi = dwi
        self.mask = mask
        self.anat = anat
        self.bvec = bvec
        self.bval = bval
        self.phasediff = phasediff

    def __str__(self):
        str_to_print = messages.INITIATEMRTRIX.format(
            mrtrix_dir=self.mrtrix_dir,
            dwi=self.dwi.name,
            anat=self.anat.name,
            phasediff=self.phasediff.name,
        )
        return str_to_print

    def transfer_files_to_mrt(self):
        files_list = [self.anat, self.dwi, self.mask, self.phasediff]
        print("Converting files to .mif format...")
        for f in files_list:
            f_name = Path(f.stem).stem + ".mif"
            new_f = Path(self.mrtrix_dir / f_name)
            if not new_f.is_file():
                if "T1" in str(f):
                    print("Importing T1 image into temporary directory")
                    new_anat = dmri_methods.convert_to_mif(f, new_f)
                elif "mask" in str(f):
                    print("Importing mask image into temporary directory")
                    new_mask = dmri_methods.convert_to_mif(f, new_f)
                else:
                    if "AP" in str(f):
                        print("Importing DWI data into temporary directory")
                        new_dwi = dmri_methods.convert_to_mif(f, new_f, bvec, bval)
                    elif "PA" in str(f):
                        print(
                            "Importing reversed phased encode data into temporary directory"
                        )
                        new_PA = dmri_methods.convert_to_mif(f, new_f)
            else:
                if "T1" in str(f):
                    new_anat = new_f
                elif "mask" in str(f):
                    new_mask = new_f
                else:
                    if "AP" in str(f):
                        new_dwi = new_f
                    elif "PA" in str(f):
                        new_PA = new_f
        self.new_anat, self.new_dwi, self.new_mask, self.new_PA = (
            new_anat,
            new_dwi,
            new_mask,
            new_PA,
        )

    def run(self):
        if not self.dir_exists:
            print("Initiate Mrtrix preprocessing directory.")
            self.mrtrix_dir.mkdir()
        else:
            print("Mrtrix preprocessing directory already exists. Continuing.")
        self.transfer_files_to_mrt()
        return self.mrtrix_dir, self.new_anat, self.new_dwi, self.new_mask, self.new_PA


class DenoiseDWI:
    def __init__(self, epi_file: Path, mask: Path):
        """
        Perform mrtrix's initial denoising.
        Arguments:
            epi_file {Path} -- [Path to dwi file]
            mask {Path} -- [Path to dwi brain mask]
        """
        self.epi_file = epi_file
        self.mask = mask
        self.out_file = Path(epi_file.parent / f"{epi_file.stem}_denoised.mif")
        self.exist = check_files_existence([self.out_file])

    def __str__(self):
        str_to_print = messages.DENOISEDWI.format(
            epi_file=self.epi_file.name,
            mask=self.mask.name,
            denoised=self.out_file.name,
        )

    def denoise(self):
        denoiser = dmri_methods.denoise_dwi(self.epi_file, self.mask, self.out_file)
        denoiser.run()

    def run(self):
        if not self.exist:
            print("Performing initial denoising procedure...")
            self.denoise()
        else:
            print("Initial denoised procedure already done. Continuing.")
        return self.out_file


class Unring:
    def __init__(self, denoised: Path):
        """
        Use Mrtrix3's tools for Gibbs rings removal
        Arguments:
            denoised {Path} -- [Initally denoised DWI image]
        """
        self.denoised = denoised
        self.out_file = Path(denoised.with_name(denoised.stem + "_degibbs.mif"))
        self.exist = check_files_existence([self.out_file])

    def __str__(self):
        str_to_print = messages.UNRING.format(
            denoised=self.denoised.name, degibbs=self.out_file.name
        )

    def unring(self):
        unring = dmri_methods.unring_dwi(self.denoised, self.out_file)
        unring.run()

    def run(self):
        if not self.exist:
            print("Performing Gibbs rings removal for DWI data")
            self.unring()
        else:
            print("Gibbs rings removal already done. Continuing.")
        return self.out_file


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


class ConvertMif2Nifti:
    def __init__(self, derivatives: Path, subj: str):
        self.subj = subj
        self.mrtrix_dir = Path(derivatives / subj / "dwi" / "Mrtrix_prep")
        self.anat = Path(derivatives / subj / "anat")
        self.dwi = self.mrtrix_dir.parent

    def __str__(self):
        str_to_print = messages.CONVERT2NIFTI.format(
            mrtrix_dir=self.mrtrix_dir, subj_dir=self.anat.parent
        )
        return str_to_print

    def convert_files(self):
        for f in self.mrtrix_dir.iterdir():
            if f.is_file():
                f_name = f.name
                if (
                    "fieldmap" in f_name or "b0" in f_name or "PA" in f_name
                ) and "mif" in f_name:
                    #                    f.unlink()
                    continue
                elif (
                    ("T1" in f_name or "vis" in f_name or "5TT" in f_name)
                    and "mif" in f_name
                    and "dwi" not in f_name
                ):
                    out_file = self.anat / f"{f.stem}{FSLOUTTYPE}"
                    if not out_file.is_file():
                        print(f"Converting {f_name} to {out_file}.")
                        dmri_methods.convert_to_mif(f, out_file)
                elif ("dwi" in f_name or "AP" in f_name) and "mif" in f_name:
                    if "preprocessed" in f_name:
                        out_file = self.dwi / f"{subj}_acq-AP_{f.stem}{FSLOUTTYPE}"
                    else:
                        out_file = self.dwi / f"{f.stem}{FSLOUTTYPE}"
                    if not out_file.is_file():
                        print(f"Converting {f_name} to {out_file}.")
                        dmri_methods.convert_to_mif(f, out_file)

    def run(self):
        print(
            "Converting Mrtrix3's .mif files to the more commonly used .nii.gz format."
        )
        self.convert_files()


class BiasCorrect:
    def __init__(self, preprocessed: Path):
        self.preprocessed = preprocessed
        self.out_file = Path(preprocessed.parent / f"{preprocessed.stem}_biascorr.mif")
        self.exist = check_files_existence([self.out_file])

    def __str__(self):
        str_to_print = messages.BIASCORRECTION.format(
            preprocessed=self.preprocessed, bias_corrected=self.out_file
        )
        return str_to_print

    def bias_correct(self):
        self.out_file = dmri_methods.bias_correct(self.preprocessed, self.out_file)

    def run(self):
        if not self.exist:
            print("Performing initial B1 bias field correction of DWIs")
            self.bias_correct()
        else:
            print("Initial B1 bias field correction already done. Continuing.")


class GenerateFiveTissue:
    def __init__(self, dwi: Path, dwi_mask: Path, anat: Path, anat_mask: Path):
        self.dwi = dwi
        self.dwi_mask = dwi_mask
        self.anat = anat
        self.anat_mask = anat_mask
        self.new_highres_mask = Path(anat.parent / f"{anat.stem}_brain_mask.mif")
        self.meanbzero = Path(dwi.parent / f"{dwi.stem}_meanbzero.mif")
        self.dwi_pseudoT1 = Path(dwi.parent / f"{dwi.stem}_pseudoT1.mif")
        self.T1_pseudobzero = Path(dwi.parent / f"{dwi.stem}_T1_pseudobzero.mif")
        self.t1_registered = Path(dwi.parent / "T1_registered.mif")
        self.t1_mask_registered = Path(dwi.parent / "T1_mask_registered.mif")
        self.five_tissues = self.t1_registered.parent / "5TT.mif"
        self.vis = self.t1_registered.parent / "vis.mif"
        self.exist_first_step = check_files_existence(
            [self.t1_registered, self.t1_mask_registered]
        )
        self.exist_second_step = check_files_existence([self.five_tissues, self.vis])

    def __str__(self):
        str_to_print = messages.FIVETISSUES.format(
            parent=self.dwi.parent,
            dwi=self.dwi,
            dwi_mask=self.dwi_mask,
            anat=self.anat,
            anat_mask=self.anat_mask,
            meanbzero=self.meanbzero,
            dwi_pseudo_t1=self.dwi_pseudoT1,
            t1_pseudo_dwi=self.T1_pseudobzero,
            t1_registered=self.t1_registered,
            t1_reg_mask=self.t1_mask_registered,
            five_tissue=self.five_tissues,
            vis=self.vis,
        )
        return str_to_print

    def register_dwi_to_t1(self):
        if not self.new_highres_mask.is_file():
            self.new_highres_mask = dmri_methods.convert_to_mif(
                self.anat_mask, self.new_highres_mask
            )
        if not self.T1_pseudobzero.is_file():
            self.meanbzero, self.dwi_pseudoT1, self.T1_pseudobzero = dmri_methods.dwi_to_T1_coreg(
                self.dwi, self.dwi_mask, self.anat, self.new_highres_mask
            )
        if not self.t1_registered.is_file():
            self.t1_registered, self.t1_mask_registered = dmri_methods.reg_dwi_to_t1(
                self.dwi,
                self.anat,
                self.dwi_pseudoT1,
                self.T1_pseudobzero,
                self.meanbzero,
                self.anat_mask,
                self.dwi_mask,
                True,
            )

    def generate_five_tissues(self):
        self.vis, self.five_tissues = dmri_methods.five_tissue(
            self.t1_registered, self.t1_mask_registered
        )

    def run(self):
        if not self.exist_first_step:
            print(
                "Performing registration of DWI image to structural one (and vice versa)."
            )
            self.register_dwi_to_t1()
        else:
            print(
                "Registration between DWI image and structural already done. Continuing."
            )
        if not self.exist_second_step:
            print(
                "Generating five-tissue-type (5TT) image (for later usage of the white-matter tissue mask)."
            )
            self.generate_five_tissues()
        else:
            print("Five-tissue-type (5TT) file already exists. Continuing.")

        return self.five_tissues, self.vis


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
        return self.highres, self.highres_mask, self.highres_brain


class PreprocessPipeline:
    def __init__(
        self,
        bids_dir: Path,
        subj: str = None,
        derivatives: Path = None,
        design: Path = DESIGN,
        skip_bids: bool = False,
    ):
        self.bids_dir = bids_dir
        self.subjects = list()
        self.design = Path(design)
        if subj:
            self.subjects.append(subj)
        else:
            self.subjects = [
                Path(cur_subj).name for cur_subj in glob.glob(f"{self.bids_dir}/sub-*")
            ]
        if not derivatives:
            derivatives = Path(self.bids_dir.parent / "derivatives")
        self.derivatives = derivatives
        self.subjects.sort()
        self.skip_bids = skip_bids

    def check_bids(self):
        bids_validator = CheckBids(self.bids_dir)
        print(bids_validator)
        bids_validator.run()

    def generate_output_directory(self, subj: str):
        set_output = GenerateOutputDirectory(self.bids_dir, self.derivatives, subj)
        logging.info(set_output)
        set_output.run()

    def print_start(self, subj: str):
        (
            anat,
            func,
            sbref,
            dwi,
            bvec,
            bval,
            phasediff,
        ) = fmri_methods.load_initial_files(self.bids_dir, subj)
        str_to_print = messages.PRINT_START.format(
            subj=subj,
            subj_dir=self.bids_dir / subj,
            anat=anat,
            func=func,
            dwi=dwi,
            bvec=bvec,
            bval=bval,
            phasediff=phasediff,
        )
        return anat, func, sbref, dwi, bvec, bval, phasediff, str_to_print

    def generate_field_map(self, subj: str, AP: Path, PA: Path):
        fieldmap_generator = GenerateFieldMap(subj, AP, PA, self.derivatives)
        logging.info(fieldmap_generator)
        fieldmap_rad, fieldmap_mag_brain, index, acq = fieldmap_generator.run()
        mask = Path(
            fieldmap_mag_brain.parent
            / f"{Path(fieldmap_mag_brain.stem).stem}_mask{FSLOUTTYPE}"
        )
        return fieldmap_rad, fieldmap_mag_brain, mask, index, acq

    def preprocess_anat(self, subj: str, anat: Path):
        anat_preprocess = StructuralPreprocessing(self.derivatives, subj, anat)
        logging.info(anat_preprocess)
        highres_head, highres_brain, highres_mask = anat_preprocess.run()
        return highres_head, highres_brain, highres_mask

    def run_feat(
        self,
        subj: str,
        epi_file: Path,
        highres: Path,
        fieldmap_brain: Path,
        fieldmap_rad: Path,
    ):
        feat = FEAT(
            subj,
            epi_file,
            highres,
            fieldmap_brain,
            fieldmap_rad,
            self.design,
            self.derivatives,
        )
        logging.info(feat)
        feat.run()

    def motion_correct(self, subj: str, dwi: Path):
        f_name = Path(dwi).name
        # new_f = Path(Path(f_name).stem).stem + f"_MotionCorrected{FSLOUTTYPE}"
        motion_corrected = self.derivatives / subj / "dwi" / f_name
        init_correction = MotionCorrection(dwi, motion_corrected)
        logging.info(init_correction)
        motion_corrected = init_correction.run()
        return motion_corrected

    def initiate_mrtrix_prep(
        self,
        subj: str,
        dwi: Path,
        mask: Path,
        anat: Path,
        bvec: Path,
        bval: Path,
        phasediff: Path,
    ):
        init_mrt = InitiateMrtrix(
            subj, self.derivatives, dwi, mask, anat, bvec, bval, phasediff
        )
        logging.info(init_mrt)
        mrt_folder, new_anat, new_dwi, new_mask, new_phasediff = init_mrt.run()
        return mrt_folder, new_anat, new_dwi, new_mask, new_phasediff

    def denoise_dwi(self, dwi: Path, mask: Path):
        denoiser = DenoiseDWI(dwi, mask)
        logging.info(denoiser)
        denoised = denoiser.run()
        return denoised

    def degibbs(self, denoised: Path):
        degibbser = Unring(denoised)
        logging.info(degibbser)
        degibbsed = degibbser.run()
        return degibbsed

    def eddy_correct(self, degibbs: Path, phasediff: Path):
        eddy_corrector = PreprocessDWI(degibbs, phasediff)
        logging.info(eddy_corrector)
        eddy_corrected = eddy_corrector.run()
        return eddy_corrected

    def bias_correct(self, preprocessed: Path):
        bias_corrector = BiasCorrect(preprocessed)
        logging.info(bias_corrector)
        bias_corrected = bias_corrector.run()
        return bias_corrected

    def generate_five_tissue(
        self, dwi: Path, dwi_mask: Path, anat: Path, anat_mask: Path
    ):
        five_tissues = GenerateFiveTissue(dwi, dwi_mask, anat, anat_mask)
        logging.info(five_tissues)
        five_tissue, vis = five_tissues.run()

    def convert_to_nifti(self, subj: str):
        converter = ConvertMif2Nifti(self.derivatives, subj)
        logging.info(converter)
        converter.run()

    def run(self):
        print("Iinitiating preprocessing procedures...")
        print(f"Input directory: {self.bids_dir}")
        print(f"Output directory: {self.derivatives}")
        if not self.skip_bids:
            self.check_bids()
        for subj in self.subjects:
            logging.basicConfig(
                filename=self.derivatives / subj / "preprocessing.log",
                filemode="w",
                format="%(asctime)s - %(message)s",
                level=logging.INFO,
            )
            print(f"Currently preprocessing {subj}'s images.'")
            t = time.time()
            self.generate_output_directory(subj)
            anat, func, sbref, dwi, bvec, bval, phasediff, str_to_print = self.print_start(
                subj
            )
            fieldmap_rad, fieldmap_mag_brain, mask, index, acq = self.generate_field_map(
                subj, dwi, phasediff
            )
            highres_head, highres_brain, highres_mask = self.preprocess_anat(subj, anat)
            if func:
                self.run_feat(
                    subj, func, highres_brain, fieldmap_mag_brain, fieldmap_rad
                )
            if dwi:
                motion_corrected = self.motion_correct(subj, dwi)
                mrt_folder, new_anat, new_dwi, new_mask, new_phasediff = self.initiate_mrtrix_prep(
                    subj, motion_corrected, mask, highres_head, bvec, bval, phasediff
                )
                denoised = self.denoise_dwi(new_dwi, new_mask)
                degibbsed = self.degibbs(denoised)
                preprocessed = self.eddy_correct(degibbsed, new_phasediff)
                bias_corrected = self.bias_correct(preprocessed)
                self.generate_five_tissue(
                    bias_corrected, new_mask, new_anat, highres_mask
                )
                self.convert_to_nifti(subj)
                elapsed = (time.time() - t) / 60
                logging.info(
                    print(
                        "%s`s preproceesing procedures took %.2f minutes."
                        % (subj, elapsed)
                    )
                )

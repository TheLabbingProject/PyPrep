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

MOTHER_DIR = Path("/Users/dumbeldore/Desktop/bids_dataset")
ATLAS = Atlases.megaatlas.value
DESIGN = Templates.design.value
FSLOUTTYPE = ".nii.gz"


class BidsPrep:
    def __init__(
        self,
        mother_dir: Path,
        seq: str,
        subj: str = None,
        out_dir: Path = None,
        atlas: Path = ATLAS,
        design: Path = DESIGN,
        skip_bids: bool = False,
    ):
        """
        Initiate the BIDS preprocessing class with either a specific subject or None = all subjects (default)
        Arguments:
            mother_dir {Path} -- [Path to a BIDS compliant directory (should contain "mother_dir/sub-xx")]
            seq (str) -- ["func","dwi","anat". Specify the modality of preprocessing you would like to produce.]
        Keyword Arguments:
            subj (str) -- ["sub-xx" for specific subject or None for all subjects] (default: {None})
            atlas (Path) -- [path to atlas` directory. should contain "Highres" and "Labels" files.] (default: '/atlases/megaatlas')
            design (Path) -- [Path to FEAT's design template.] (default: {'/templates/design.fsf'})
        """
        self.mother_dir = Path(mother_dir)
        self.subjects = list()
        self.seq = seq
        self.design = Path(design)
        if subj:
            self.subjects.append(subj)
        else:
            self.subjects = [
                Path(cur_subj).name
                for cur_subj in glob.glob(f"{self.mother_dir}/sub-*")
            ]
        if not out_dir:
            out_dir = Path(self.mother_dir.parent / "derivatives")
        self.out_dir = out_dir
        self.atlas = Path(atlas)
        self.subjects.sort()
        if not atlas:
            self.highres_atlas, self.labels_atlas = self.get_atlas()
        else:
            for f in atlas.iterdir():
                if "highres.nii" in str(f):
                    self.highres_atlas = f
                elif "Labels.nii" in str(f):
                    self.labels_atlas = f
        self.skip_bids = skip_bids

    def get_atlas(self):
        """
        GUI-based request from user to guide to atlas directory (megaatlas)
        """
        if not self.atlas:
            root = tk.Tk()
            root.withdraw()
            atlas = Path(
                filedialog.askdirectory(
                    title="Please choose the folder containing your megaatlas files."
                )
            )
        for f in atlas.iterdir():
            if "highres.nii" in str(f):
                highres_atlas = f
            elif "Labels.nii" in str(f):
                labels_atlas = f
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
        in_name = self.mother_dir.name
        out_name = self.out_dir.name
        dirs = glob.glob(f"{Path(self.mother_dir / subj)}/*")
        dirs.append(f"{os.path.dirname(dirs[0])}/atlases")
        dirs.append(f"{os.path.dirname(dirs[0])}/scripts")
        for d in dirs:
            new_d = d.replace(in_name, out_name)
            if not os.path.isdir(new_d):
                os.makedirs(new_d)
        print(f"Sorted output directory to resemble input, BIDS compatible one:")
        fmri_methods.list_files(str(self.out_dir))

    def check_file(self, in_file: Path):
        """
        Checks wether a file exists
        Arguments:
            in_file {Path} -- [path of file to check]

        Returns:
            [bool] -- [True or False]
        """
        in_file = Path(in_file)
        return not in_file.is_file()

    def generate_field_map(self, subj: str, AP: Path, PA: Path):
        """
        Generate Field Maps using dwi's AP and phasediff's PA scans
        Arguments:
            AP {Path} -- [path to dwi's AP file]
            PA {Path} -- [path to PA phasediff file]
            outdir {Path} -- [path to output directory (outdir/sub-xx/fmap)]
        """
        outdir = self.out_dir / subj / "fmap"
        merged = outdir / f"merged_phasediff{FSLOUTTYPE}"
        if not merged.exists():
            print("generating dual-phase encoded image...")
            merger = fmri_methods.MergePhases(AP, PA, merged)
            merger.run()
        else:
            print("Dual-phase encoded image already exists...")
        datain = outdir / "datain.txt"
        if not datain.exists():
            print("Generating datain.txt with dual-phase data...")
            fmri_methods.GenDatain(AP, PA, datain)
        else:
            print("datain.txt already exists...")
        index_file = outdir / "index.txt"
        if not index_file.exists():
            print("Generating index.txt file, for later eddy-currents correction...")
            dmri_methods.GenIndex(AP, index_file)
        else:
            print("index.txt file already exists...")
        fieldmap = outdir / f"fieldmap{FSLOUTTYPE}"
        fieldmap_rad = Path(
            fieldmap.parent / f"{Path(fieldmap.stem).stem}_rad{FSLOUTTYPE}"
        )
        fieldmap_mag = Path(
            fieldmap.parent / f"{Path(fieldmap.stem).stem}_magnitude{FSLOUTTYPE}"
        )
        if not fieldmap_mag.exists():
            print("Using TopUp method to generate fieldmap images...")
            fieldmap_mag, fieldmap_rad = fmri_methods.TopUp(merged, datain, fieldmap)
        else:
            print("Fieldmap images already exists...")
        fieldmap_brain = self.BrainExtract(subj, fieldmap_mag, "fmap")
        return fieldmap_rad, fieldmap_brain, index_file, datain

    def BrainExtract(self, subj: str, in_file: Path, seq: str = "anat"):
        """
        Perform brain extraction using FSL's bet.
        *** Need improvement - maybe use cat12
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]
            in_file {Path} -- [Path to file to perform brain extraction on.]
        Keyword Arguments:
            subdir {str} -- ['anat','dwi' or 'func' perform brain extraction for an image in specific subdirectory in mother_dir.] (default: {'anat'})
        """
        in_file = Path(in_file)
        f_name = Path(in_file.name)
        new_f = Path(f_name.stem).stem + f"_brain{FSLOUTTYPE}"
        if seq == "anat":
            shutil.copy(in_file, Path(self.out_dir / subj / seq))
        out_file = Path(self.out_dir / subj / seq / new_f)
        if self.check_file(out_file):
            print("Performing brain extractoin using FSL`s bet")
            print(f"Input file: {in_file}")
            print(f"Output file: {out_file}")
            bet = fmri_methods.BetBrainExtract(in_file, out_file)
            bet.run()
        else:
            print("Brain extraction already done.")
            print(f"Brain-extracted file is at {out_file}")
        return out_file

    def MotionCorrect(self, subj: str, in_file: Path, seq: str = "func"):
        """
        Perform motion correction with FSL's MCFLIRT
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]
            in_file {Path} -- [Path to a 4D file to perform motion correction (realignment) on.]

        Keyword Arguments:
            seq {str} -- ['func' or 'dwi', Modality to perform motion correction on.] (default: {"func"})
        """

        f_name = Path(in_file).name
        new_f = Path(Path(f_name).stem).stem + f"_MotionCorrected{FSLOUTTYPE}"
        out_file = self.out_dir / subj / seq / new_f
        img = nib.load(in_file)
        is_4d = img.ndim > 3
        if is_4d:
            if self.check_file(out_file):
                mot_cor = fmri_methods.MotionCorrect(in_file=in_file, out_file=out_file)
                print("Performing motion correction using FSL`s MCFLIRT")
                print(f"Input file: {in_file}")
                print(f"Output file: {out_file}")
                mot_cor.run()
            else:
                print("Motion correction already done.")
                print(f"Motion-corrected file is at {out_file}")
        else:
            print(
                "Input files isn't a 4D image, therefore isn't elligable for motion correction."
            )
        Path(out_file).rename(in_file)

        return in_file

    def prep_feat(
        self,
        subj: str,
        epi_file: Path,
        highres: Path,
        fieldmap_mag_brain: Path,
        fieldmap_rad: Path,
    ):
        """
        Perform FSL's feat preprocessing
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]
            epi_file {str} -- [Path to a 4D file use as input for FSL's feat.]
            highres {str} -- [Path to brain-extracted ('_brain') highres structural file]
            fieldmap_rad {Path} -- [Path tp fieldmap file in radians]
            fieldmap_mag_brain {Path} -- [Path to fieldmap magnitude brain extracted image]
        Returns:
            [str] -- [Path to .feat directory]
        """
        temp_design = self.design
        subj_design = self.out_dir / subj / "scripts" / "func_design.fsf"
        out_feat = self.out_dir / subj / "func" / "func.feat"
        GenFsf = fmri_methods.FeatDesign(
            out_feat,
            epi_file,
            highres,
            temp_design,
            subj_design,
            fieldmap_rad,
            fieldmap_mag_brain,
        )
        fsf = GenFsf.run()
        if not os.path.isdir(out_feat):
            print(f"Running FEAT for {subj}")
            print(f"input: {epi_file.name}")
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
        flt = fmri_methods.FLIRT(in_file, ref, out_dir, proc)
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
        fmri_methods.list_files(f"{self.out_dir}/{subj}")

    def load_transforms(self, feat: str):
        reg_dir = Path(f"{feat}/reg")
        for f in os.listdir(reg_dir):
            if "highres2example_func.mat" in f:
                highres2func_aff = str(reg_dir / f)
            elif "standard2highres.mat" in f:
                standard2highres_aff = str(reg_dir / f)
            elif f == "highres.nii.gz":
                highres = str(reg_dir / f)
        epi_file = str(Path(feat) / "mean_func.nii.gz")
        return epi_file, highres, highres2func_aff, standard2highres_aff

    def generate_atlas(
        self,
        subj: str,
        epi_file: str,
        highres: str,
        highres2func: str,
        standard2highres: str,
    ):

        out_dir = str(Path(self.out_dir) / subj / "atlases")
        atlas_in_highres = str(Path(out_dir) / "atlas2subj_highres_Linear.nii")
        flt1 = fmri_methods.FLIRT(
            self.highres_atlas, highres, out_dir, standard2highres
        )
        flt1.run()

        atlas_in_epi = str(Path(out_dir) / "atlas2subj_epi_Linear.nii")
        flt2 = fmri_methods.lin_atlas2subj(
            atlas_in_highres, epi_file, out_dir, highres2func
        )
        flt2.run()

    def params_for_eddy(self, subj):
        """
        Get paths for eddy-required files
        Arguments:
            subj {[type]} -- ['sub-xx' in a BIDS compatible directory]
        """
        fmap_dir: Path = self.out_dir / subj / "fmap"
        for f in fmap_dir.iterdir():
            f = str(f)
            if "index.txt" in f:
                index = Path(f)
            elif "fieldcoef.nii" in f:
                fieldcoef = Path(f)
            elif "movpar.txt" in f:
                movpar = Path(f)
            elif "mask.nii" in f:
                mask = Path(f)
        return fieldcoef, movpar, mask

    def run_eddy(
        self,
        subj: str,
        epi_file: Path,
        mask: Path,
        index: Path,
        acq: Path,
        bvec: Path,
        bval: Path,
        fieldcoef: Path,
        movpar: Path,
    ):
        """
        Generating FSL's eddy-currents correction tool, eddy, with specific inputs.
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]
            epi_file {Path} -- [Path to dwi file]
            mask {Path} -- [Path to brain mask file]
            index {Path} -- [Path to index.txt file]
            acq {Path} -- [Path to datain.txt file]
            bvec {Path} -- [Path to bvec file (extracted automatically when converting the dicom file)]
            bval {Path} -- [Path to bval file (extracted automatically when converting the dicom file)]
            fieldcoef {Path} -- [Path to field coeffient as extracted after topup procedure]
            movpar {Path} -- [Path to moving parameters as extracted after topup procedure]
        Returns:
            eddy [type] -- [nipype's FSL eddy's interface, generated with specific inputs]
        """
        out_base = self.out_dir / subj / "dwi" / "eddy_corrected"
        eddy = dmri_methods.eddy_correct(
            epi_file, mask, index, acq, bvec, bval, fieldcoef, movpar, out_base
        )
        print("Running FSL's eddy-current correction tool:")
        print(eddy.cmdline)
        eddy.run()

    def initiate_mrtrix_dir(self, subj: str):
        """
        Initiate a directory for Mrtrix3 preprocessing outputs
        Arguments:
            subj {str} -- ['sub-xx' in a BIDS compatible directory]

        Returns:
            [Path] -- [Path to subjects' Mrtrix preprocessing directory]
        """
        mrtrix_dir = Path(self.out_dir / subj / "dwi" / "Mrtrix_prep")
        if not mrtrix_dir.is_dir():
            mrtrix_dir.mkdir()
            print(f"Generated mrtrix prep directory at {mrtrix_dir}")
        return mrtrix_dir

    def transfer_files_to_mrt(
        self,
        mrt_folder: Path,
        dwi: Path,
        mask: Path,
        anat: Path,
        bvec: Path,
        bval: Path,
        phasediff: Path,
    ):
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
        files_list = [anat, dwi, mask, phasediff]
        print("Converting files to .mif format...")
        for f in files_list:
            f_name = Path(f.stem).stem + ".mif"
            new_f = Path(mrt_folder / f_name)
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
        return new_anat, new_dwi, new_mask, new_PA

    def dwi_denoise(self, epi_file: Path, mask: Path):
        """
        Perform mrtrix's initial denoising.
        Arguments:
            epi_file {Path} -- [Path to dwi file]
            mask {Path} -- [Path to dwi brain mask]
        """
        out_file = Path(epi_file.parent / f"{epi_file.stem}_denoised.mif")
        if not out_file.is_file():
            print("Performing initial denoising procedure...")
            denoise = dmri_methods.Denoise(epi_file, mask, out_file)
            denoise.run()
        return out_file

    def unring(self, denoised: Path):
        out_file = Path(denoised.with_name(denoised.stem + "_degibbs.mif"))
        if not out_file.is_file():
            print("Performing Gibbs ringing removal for DWI data")
            unring = dmri_methods.Unring(denoised, out_file)
            unring.run()
        return out_file

    def DWI_preproc(self, degibbs: Path, phasediff: Path):
        out_file = Path(degibbs.parent / "dwi_preprocessed.mif")
        if not out_file.is_file():
            print("Performing various geometric corrections of DWIs")
            cmd = dmri_methods.DWI_prep(degibbs, phasediff, out_file)
            os.system(cmd)
        return out_file

    def run(self):
        """
        Complete preprocessing procedure
        """
        print("Iinitiating preprocessing procedures...")
        print(f"Input directory: {self.mother_dir}")
        print(f"Output directory: {self.out_dir}")
        if not self.skip_bids:
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
            ) = fmri_methods.load_initial_files(self.mother_dir, subj)
            fieldmap_rad, fieldmap_mag, index, acq = self.generate_field_map(
                subj, dwi, phasediff
            )
            mask = Path(
                fieldmap_mag.parent / f"{Path(fieldmap_mag.stem).stem}_mask{FSLOUTTYPE}"
            )
            fieldcoef, movpar, mask = self.params_for_eddy(subj)
            highres_brain = self.BrainExtract(subj, anat)
            if func:
                feat = self.prep_feat(
                    subj, func, highres_brain, fieldmap_mag, fieldmap_rad
                )
                (
                    epi_file,
                    highres,
                    highres2func,
                    standard2highres,
                ) = self.load_transforms(feat)
                # self.generate_atlas(
                #     subj, epi_file, highres, highres2func, standard2highres
                # )
            if dwi:
                mrt_folder = self.initiate_mrtrix_dir(subj)
                motion_corrected = self.MotionCorrect(subj, dwi, "dwi")
                new_anat, new_dwi, new_mask, new_phasediff = self.transfer_files_to_mrt(
                    mrt_folder, motion_corrected, mask, anat, bvec, bval, phasediff,
                )
                denoised = self.dwi_denoise(new_dwi, new_mask)
                degibbs = self.unring(denoised)
                dwi_preprocessed = self.DWI_preproc(degibbs, new_phasediff)
                # self.run_eddy(
                #    subj, dwi, mask, index, acq, bvec, bval, fieldcoef, movpar
                # )
            elapsed = (time.time() - t) / 60
            print("%s`s preproceesing procedures took %.2f minutes." % (subj, elapsed))


if __name__ == "__main__":
    bids_dir = Path("/home/gal/bids_dataset")
    t = BidsPrep(mother_dir=bids_dir, subj="sub-01", seq="func", skip_bids=True)
    t.run()

# to run ICA-AROMA:
# 1. bet T1
# 2. epi-reg from rest to t1 (.mat)
# 3. fnirt from t1 to mni (.nii.gz)
# 4. mcflirt rest (.par)

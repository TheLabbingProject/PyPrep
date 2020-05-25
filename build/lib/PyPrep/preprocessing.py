from PyPrep.utils import (
    BrainExtraction,
    CheckBids,
    DisplayablePath,
    GenerateFiveTissue,
    GenerateOutputDirectory,
    MotionCorrection,
    check_files_existence,
    FSLOUTTYPE,
)
from PyPrep.anatomy import StructuralPreprocessing
from PyPrep.functional import FEAT, fmri_prep_functions as fmri_methods
from PyPrep.diffusion import (
    BiasCorrect,
    ConvertMif2Nifti,
    PreprocessDWI,
    DenoiseDWI,
    GenerateFieldMap,
    InitiateMrtrix,
    Unring,
    dmri_prep_functions as dmri_methods,
)

import glob
import nibabel as nib
import os
import shutil
import subprocess
import time
from pathlib import Path
from atlases.atlases import Atlases
from templates.templates import Templates
from logs.messages import PRINT_START
import logging

ATLAS = Atlases.megaatlas.value
DESIGN = Templates.design.value
# FSLOUTTYPE = ".nii.gz"

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
        str_to_print = PRINT_START.format(
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
        init_correction = MotionCorrection(dwi, dwi)
        logging.info("Performing motion correction for DWI image.")
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
            print(f"Currently preprocessing {subj}'s images.'")
            t = time.time()
            self.generate_output_directory(subj)
            (
                anat,
                func,
                sbref,
                dwi,
                bvec,
                bval,
                phasediff,
                str_to_print,
            ) = self.print_start(subj)
            logging.basicConfig(
                filename=self.derivatives / subj / "preprocessing.log",
                filemode="w",
                format="%(asctime)s - %(message)s",
                level=logging.INFO,
            )
            logging.info(str_to_print)
            (
                fieldmap_rad,
                fieldmap_mag_brain,
                mask,
                index,
                acq,
            ) = self.generate_field_map(subj, dwi, phasediff)
            highres_head, highres_brain, highres_mask = self.preprocess_anat(subj, anat)
            if func:
                self.run_feat(
                    subj, func, highres_brain, fieldmap_mag_brain, fieldmap_rad
                )
            if dwi:
                motion_corrected = self.motion_correct(subj, dwi)
                (
                    mrt_folder,
                    new_anat,
                    new_dwi,
                    new_mask,
                    new_phasediff,
                ) = self.initiate_mrtrix_prep(
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


if __name__ == "__main__":
    bids_dir = Path("/home/gal/bids_dataset")
    derivatives = Path("/home/gal/derivatives_2")
    bids_prep = PreprocessPipeline(bids_dir, derivatives=derivatives, skip_bids=True)
    bids_prep.run()

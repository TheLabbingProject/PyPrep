from nipype.interfaces import fsl
import nibabel as nib
from pathlib import Path
import os
from nilearn.image import index_img
import sys
import subprocess
import json
import numpy as np

FSLOUTTYPE = ".nii.gz"


def check_nifti_output():
    info = fsl.Info
    cur_output = info.output_type()
    fsldir = Path(os.getenv("FSLDIR"))
    fslconf_dir = Path(Path.home() / ".fslconf")
    fslconf_file = fslconf_dir / "fsl.sh"
    if cur_output != "NIFTI_GZ":
        if not fslconf_dir.exists():
            fslconf_dir.mkdir()
        with fslconf_file.open(mode="w+") as conf_file:
            conf_file.write("FSLOUTTYPE=NIFTI_GZ\n")
            conf_file.write("export FSLOUTTYPE")
            os.system("export FSLOUTTYPE=NIFTI_GZ")
        print(
            f"""FSLOUTTYPE is now set to NIFTI_GZ (.nii.gz).
        If you wish to change it in the future:
            1. Open the file located at: {fslconf_file}
            2. Change the FSLOUTTYPE variable to the desired output format."""
        )
    else:
        print("""FSLOUTTYPE is set to NIFTI_GZ (.nii.gz)""")


def load_initial_files(mother_dir: Path, sub: str):
    """
    Initiate relevant files for preprocessing procedures. Note that {mother_dir} must be a BIDS compatible directory.
    Arguments:
        mother_dir {str} -- [Path to a BIDS compliant directory (should contain "mother_dir/sub-xx")]
        sub {str} -- ['sub-xx']

    Returns:
        [type] -- [anat, func, sbref, dwi, bvec, bval, phasediff files from subject's directory.]
    """
    folder_name = mother_dir / sub
    dwi_folder = folder_name / "dwi"
    anat_folder = folder_name / "anat"
    fmap_folder = folder_name / "fmap"
    func_folder = folder_name / "func"
    anat_file = func_file = sbref_file = dwi_file = bvec = bval = phasediff = None
    for file in dwi_folder.iterdir():
        # file = str(file)
        if "dwi" in str(file):
            if "dwi.nii" in str(file):
                dwi_file = file
            elif "bvec" in str(file):
                bvec = file
            elif "bval" in str(file):
                bval = file
    for file in func_folder.iterdir():
        # file = str(file)
        if "sbref.nii" in str(file):
            sbref_file = file
        elif "bold.nii" in str(file):
            func_file = file
    for file in anat_folder.iterdir():
        # file = str(file)
        if "T1w.nii" in str(file):
            anat_file = file
    for file in fmap_folder.iterdir():
        # file = str(file)
        if "PA_epi.nii" in str(file):
            phasediff = file

    return (anat_file, func_file, sbref_file, dwi_file, bvec, bval, phasediff)


check_nifti_output()


def list_files(startpath: str):
    """
        Print a summary of directory's tree
        Arguments:
            startpath {str} -- A path to a directory to inspect
        """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            if ".DS_Store" not in f:
                print("{}{}".format(subindent, f))

def bet_brain_extract(in_file: Path, out_file: Path = None):
    """
    Perform brain extraction using FSL's BET.
    Arguments:
        in_file {Path} -- [Path to input nifti image]
    Keyword Arguments:
        out_file {Path} -- [Path to output nifti image] (default: {None})

    Returns:
        bet [nipype.interfaces.fsl.preprocess.BET] -- [nipype's implementation of FSL's bet]
    """
    in_file = Path(in_file)
    img = nib.load(in_file)
    functional = img.ndim > 3  # check if input is 4D (fmri/dti)
    if not out_file:
        f_name = Path(
            in_file.parent / Path(in_file.stem).stem
        )  # clear .nii.gz and .nii
        out_file = Path(f"{f_name}_brain{FSLOUTTYPE}")
        mask_file = Path(f"{f_name}_brain_mask{FSLOUTTYPE}")
    else:
        f_name = Path(out_file.parent / Path(out_file.stem).stem)
        mask_file = Path(f"{f_name}_mask{FSLOUTTYPE}")
    # initiate fsl
    bet = fsl.BET()
    bet.inputs.in_file = in_file
    bet.inputs.out_file = out_file
    bet.inputs.mask = True
    bet.mask_file = mask_file
    if functional:
        bet.inputs.functional = True  # perform bet on all frames
    else:
        bet.inputs.robust = True  # better outcome for structural images
    return bet, out_file    


def motion_correct(in_file: Path, out_file: Path):
    """
    Perform motion correction using FSL's MCFLIRT
    Arguments:
        in_file {Path} -- [4D nifti path]
        out_file {Path} -- [4D nifti output path]
    """
    mcflt = fsl.MCFLIRT()
    mcflt.inputs.in_file = in_file
    mcflt.inputs.out_file = out_file
    return mcflt


def merge_phases(in_file: Path, phasediff: Path, merged: Path):
    """
    Combine two images into one 4D file
    Arguments:
        in_file {Path} -- [path to 4D, AP oriented file]
        phasediff {Path} -- [path to phase-different, PA oriented file]
    """
    AP_b0 = index_img(str(in_file), 0)
    AP_file = merged.parent / f"AP_b0{FSLOUTTYPE}"
    nib.save(AP_b0, str(AP_file))
    merger = fsl.Merge()
    merger.inputs.in_files = [AP_file, phasediff]
    merger.inputs.dimension = "t"
    merger.inputs.merged_file = merged
    return merger

def generate_index(epi_file: Path, index_file: Path):
    """
    Generates index.txt file, needed to run eddy-currents corrections.
    Arguments:
        epi_file {Path} -- [Path to dwi file]
        index_file {Path} -- [Path to output index.txt file]
    """
    img = nib.load(epi_file)
    n_frames = img.shape[3]
    with open(index_file, "w") as in_file:
        for i in range(n_frames):
            in_file.write("1\n")
        in_file.close()
def generate_datain(AP: Path, PA: Path, datain: Path):
    """
    Generate datain.txt file for topup
    Arguments:
        AP {Path} -- [path to AP encoded file]
        PA {Path} -- [path to PA encoded file]
        datain {Path} -- [path to output datain.txt file]
    """
    AP, PA = [
        Path(AP.parent / f"{Path(AP.stem).stem}.json"),
        Path(PA.parent / f"{Path(PA.stem).stem}.json"),
    ]
    total_readout: list = []
    for i, f in enumerate([AP, PA]):
        echo_spacing, cur_total_readout, TE = Calculate_b0_params(f)
        total_readout.append(cur_total_readout)
    with open(datain, "w+") as out_file:
        out_file.write(f"0 -1 0 {total_readout[0]}\n0 1 0 {total_readout[1]}")


def Calculate_b0_params(json_file: Path):
    """
    Calculate effective echo spacing and total readout time from dwi's json.
    For explanations, see https://lcni.uoregon.edu/kb-articles/kb-0003
    Arguments:
        json_file {Path} -- [Path to dwi's json file]

    Returns:
        effective_spacing [float] -- [1/(BandwidthPerPixelPhaseEncode * ReconMatrixPE)]
        total_readout [float] -- [(ReconMatrixPE - 1) * effective_spacing]
    """
    with open(json_file, "r") as f:
        data = json.load(f)
        effective_spacing = data["EffectiveEchoSpacing"]
        total_readout = data["TotalReadoutTime"]
        TE = data["EchoTime"]
    return effective_spacing, total_readout, TE


def top_up(merged: Path, datain: Path, fieldmap: Path):
    """
    Generate Fieldmap
    Arguments:
        merged {Path} -- [merged AP and PA file]
        datain {Path} -- [phase encoding .txt file]
        fieldmap {Path} -- [output fieldmap file]
    """
    unwarped = Path(
        fieldmap.parent / f"{Path(fieldmap.stem).stem}_unwarped{FSLOUTTYPE}"
    )
    cmd = (
        f"topup --imain={merged} --datain={datain} --fout={fieldmap} --iout={unwarped}"
    )
    os.system(cmd)
    fieldmap_rads = Path(
        fieldmap.parent / f"{Path(fieldmap.stem).stem}_rad{FSLOUTTYPE}"
    )
    cmd = f"fslmaths {fieldmap} -mul 6.28 {fieldmap_rads}"
    os.system(cmd)
    fieldmap_mag = Path(
        fieldmap.parent / f"{Path(fieldmap.stem).stem}_magnitude{FSLOUTTYPE}"
    )
    cmd = f"fslmaths {unwarped} -Tmean {fieldmap_mag}"
    os.system(cmd)

class FeatDesign:
    def __init__(
        self,
        out_dir: Path,
        in_file: Path,
        highres_brain: Path,
        temp_design: Path,
        out_design: Path,
        fieldmap_rad: Path,
        fieldmap_brain: Path,
    ):
        self.out_dir = out_dir
        self.epi_file = in_file
        self.highres = highres_brain
        self.temp_design = temp_design
        self.subj_design = out_design
        self.fieldmap_rad = fieldmap_rad
        self.fieldmap_brain = fieldmap_brain

    def init_params(self):
        epi_json = Path(f"{self.epi_file.parent}/{Path(self.epi_file.stem).stem}.json")
        effective_spacing, total_readout, TE = Calculate_b0_params(epi_json)
        effective_spacing = effective_spacing * 1000
        TE = TE * 1000
        img = nib.load(self.epi_file)
        IMG_size = str(np.sum(img.get_fdata() > 0))
        TR = str(img.header.get_zooms()[3])
        ntime = str(img.shape[-1])

        return TR, IMG_size, ntime, TE, effective_spacing

    def GenReplacements(self, IMG_size, TR, ntime, TE, effective_spacing):
        replacements = {
            "n_frames": ntime,
            "outdir": self.out_dir,
            "cur_TR": TR,
            "epi_file": self.epi_file,
            "highres_brain": self.highres,
            "n_voxels": IMG_size,
            "cur_TE": TE,
            "fieldmap_rad": self.fieldmap_rad,
            "fieldmap_mag_brain": self.fieldmap_brain,
            "effective_spacing": effective_spacing,
        }
        return replacements

    def GenFsf(self, replacements, subj_fsf):
        with open(self.temp_design) as infile:
            with open(self.subj_design, "w") as out_file:
                for line in infile:
                    for src, target in replacements.items():
                        line = line.replace(src, str(target))
                    out_file.write(line)
        print(f"Created fsf at {self.subj_design}")

    def run(self):
        TR, n_voxels, n_frames, TE, effective_spacing = self.init_params()
        replacements = self.GenReplacements(
            n_voxels, TR, n_frames, TE, effective_spacing
        )
        self.GenFsf(replacements, self.subj_design)
        return self.subj_design

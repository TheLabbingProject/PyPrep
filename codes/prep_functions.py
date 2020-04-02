from nipype.interfaces import fsl
import nibabel as nib
from pathlib import Path
import os
from nilearn.image import index_img
import sys
import subprocess
import json


def check_nifti_output():
    info = fsl.Info
    cur_output = info.output_type()
    fsldir = Path(os.getenv("FSLDIR"))
    fslconf_dir = Path(Path.home() / ".fslconf")
    fslconf_file = fslconf_dir / "fsl.sh"
    if cur_output != "Nifti":
        if not fslconf_dir.exists():
            fslconf_dir.mkdir()
        with fslconf_file.open(mode="w+") as conf_file:
            conf_file.write("FSLOUTTYPE=NIFTI\n")
            conf_file.write("export FSLOUTTYPE")
        print(
            f"""FSLOUTTYPE is now set to NIFTI (.nii).
        If you wish to change it in the future:
            1. Open the file located at: {fslconf_file}
            2. Change the FSLOUTTYPE variable to the desired output format."""
        )


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


def BetBrainExtract(in_file: Path, out_file: Path = None):
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
        out_file = Path(f"{f_name}_brain.nii")
        mask_file = Path(f"{f_name}_brain_mask.nii")
    else:
        f_name = Path(out_file.parent / Path(out_file.stem).stem)
        mask_file = Path(f"{f_name}_mask.nii")
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
    return bet


def MotionCorrect(in_file: Path, out_file: Path):
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


def MergePhases(in_file: Path, phasediff: Path, merged: Path):
    """
    Combine two images into one 4D file
    Arguments:
        in_file {Path} -- [path to 4D, AP oriented file]
        phasediff {Path} -- [path to phase-different, PA oriented file]
    """
    AP_b0 = index_img(str(in_file), 0)
    AP_file = merged.parent / "AP_b0.nii"
    nib.save(AP_b0, str(AP_file))
    merger = fsl.Merge()
    merger.inputs.in_files = [AP_file, phasediff]
    merger.inputs.dimension = "t"
    merger.inputs.merged_file = merged
    return merger


def GenDatain(AP: Path, PA: Path, datain: Path):
    """
    Generate datain.txt file for topup
    Arguments:
        AP {Path} -- [path to AP encoded file]
        PA {Path} -- [path to PA encoded file]
        datain {Path} -- [path to output datain.txt file]
    """
    AP, PA = [AP.with_suffix(".json"), PA.with_suffix(".json")]
    total_readout: list = []
    for i, f in enumerate([AP, PA]):
        with open(f, "r") as json_file:
            data = json.load(json_file)
        total_readout.append(data["BandwidthPerPixelPhaseEncode"])
    with open(datain, "w+") as out_file:
        out_file.write(f"0 -1 0 {total_readout[0]}\n0 1 0 {total_readout[1]}")


def TopUp(merged: Path, datain: Path, fieldmap: Path):
    """
    Generate Fieldmap
    Arguments:
        merged {Path} -- [merged AP and PA file]
        datain {Path} -- [phase encoding .txt file]
        fieldmap {Path} -- [output fieldmap file]
    """
    unwarped = fieldmap.with_name("unwarped.nii")
    cmd = (
        f"topup --imain={merged} --datain={datain} --fout={fieldmap} --iout={unwarped}"
    )
    os.system(cmd)
    fieldmap_rads = fieldmap.with_name(fieldmap.stem + "_rad.nii")
    cmd = f"fslmaths {fieldmap} -mul 6.28 {fieldmap_rads}"
    os.system(cmd)
    fieldmap_mag = fieldmap.with_name(fieldmap.stem + "_magnitude.nii")
    cmd = f"fslmaths {unwarped} -Tmean {fieldmap_mag}"
    os.system(cmd)
    return fieldmap_mag, fieldmap_rads

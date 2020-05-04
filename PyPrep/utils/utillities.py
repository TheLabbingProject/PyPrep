from pathlib import Path
import nibabel as nib
import nipype.interfaces.fsl as fsl
import nipype.interfaces.mrtrix3 as mrt
import os


FSLOUTTYPE = ".nii.gz"


def five_tissue(t1_registered: Path, t1_mask_registered: Path):
    """
    Generate 5TT (5-tissue-type) image based on the registered T1 image.
    Arguments:
        t1_registered {Path} -- [T1 registered to DWI space]
        t1_mask_registered {Path} -- [T1 mask]

    Returns:
        [type] -- [5TT and vis files]
    """
    out_file = Path(t1_registered.parent / "5TT.mif")
    out_vis = Path(t1_registered.parent / "vis.mif")
    if not out_file.is_file():
        seg = mrt.Generate5tt()
        seg.inputs.in_file = t1_registered
        seg.inputs.out_file = out_file
        seg.inputs.algorithm = "fsl"
        cmd = seg.cmdline + f" -mask {t1_mask_registered}"
        print(cmd)
        seg.run()
    if not out_vis.is_file():
        os.system(f"5tt2vis {str(out_file)} {str(out_vis)}")
    return out_vis, out_file


def dwi_to_T1_coreg(dwi_file: Path, dwi_mask: Path, t1_file: Path, t1_mask: Path):
    parent = dwi_file.parent
    meanbzero = parent / f"{dwi_file.stem}_meanbzero.mif"
    # meanbzero = dwi_file.replace(".mif", "_meanbzero.mif")
    cmd = f"dwiextract {dwi_file} -bzero - | mrcalc - 0.0 -max - | mrmath - mean -axis 3 {meanbzero}"
    print(cmd)
    os.system(cmd)
    dwi_pseudoT1 = parent / f"{dwi_file.stem}_pseudoT1.mif"
    # dwi_pseudoT1 = dwi_file.replace(".mif", "_pseudoT1.mif")
    cmd = f"mrcalc 1 {meanbzero} -div {dwi_mask} -mult - | mrhistmatch nonlinear - {t1_file} {dwi_pseudoT1} -mask_input {dwi_mask} -mask_target {t1_mask}"
    print(cmd)
    os.system(cmd)
    T1_pseudobzero = parent / "T1_pseudobzero.mif"
    # T1_pseudobzero = f"{os.path.dirname(dwi_file)}/T1_pseudobzero.mif"
    cmd = f"mrcalc 1 {t1_file} -div {t1_mask} -mult - | mrhistmatch nonlinear - {meanbzero} {T1_pseudobzero} -mask_input {t1_mask} -mask_target {dwi_mask}"
    print(cmd)
    os.system(cmd)
    return meanbzero, dwi_pseudoT1, T1_pseudobzero


def reg_dwi_to_t1(
    dwi_file: Path,
    t1_brain: Path,
    dwi_pseudoT1: Path,
    T1_pseudobzero: Path,
    meanbzero: Path,
    t1_mask: Path,
    dwi_mask: Path,
    run: bool,
):
    working_dir = dwi_file.parent
    t1_registered = working_dir / "T1_registered.mif"
    t1_mask_registered = working_dir / "T1_mask_registered.mif"
    if run:
        rig_t1_to_pseudoT1 = working_dir / "rigid_T1_to_pseudoT1.txt"
        rig_t1_to_dwi = working_dir / "rigid_T1_to_dwi.txt"
        rig_pseudob0_to_b0 = working_dir / "rigid_pseudobzero_to_bzero.txt"
        cmd_1 = f"mrregister {t1_brain} {dwi_pseudoT1} -type rigid -mask1 {t1_mask} -mask2 {dwi_mask} -rigid {rig_t1_to_pseudoT1}"
        cmd_2 = f"mrregister {T1_pseudobzero} {meanbzero} -type rigid -mask1 {t1_mask} -mask2 {dwi_mask} -rigid {rig_pseudob0_to_b0}"
        cmd_3 = f"transformcalc {rig_t1_to_pseudoT1} {rig_pseudob0_to_b0} average {rig_t1_to_dwi}"
        cmd_4 = f"mrtransform {t1_brain} {t1_registered} -linear {rig_t1_to_dwi}"
        cmd_5 = f"mrtransform {t1_mask} {t1_mask_registered} -linear {rig_t1_to_dwi} -template {t1_registered} -interp nearest -datatype bit"
        for cmd in [cmd_1, cmd_2, cmd_3, cmd_4, cmd_5]:
            print(cmd)
            os.system(cmd)
    return t1_registered, t1_mask_registered


def convert_to_mif(in_file, out_file, bvec=None, bval=None, anat=False):
    mrconvert = mrt.MRConvert()
    mrconvert.inputs.in_file = in_file
    mrconvert.inputs.out_file = out_file
    mrconvert.inputs.args = "-quiet"
    if bvec and bval:
        mrconvert.inputs.args = (
            f"-json_import {mrconvert.inputs.in_file.replace('nii.gz','json')}"
        )
        mrconvert.inputs.grad_fsl = (bvec, bval)
    if anat:
        mrconvert.inputs.args = "-strides +1,+2,+3 -quiet"
    mrconvert.run()
    return out_file


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




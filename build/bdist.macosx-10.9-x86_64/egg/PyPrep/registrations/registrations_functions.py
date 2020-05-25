from nipype.interfaces import fsl
from pathlib import Path
import nibabel as nib
import numpy as np


def invert_warp(in_file: Path, ref: Path, out_warp: Path, in_warp: Path):
    """
    Apply FSL's invwarp to invert a warp file
    Arguments:
        in_file {Path} -- [description]
        out_file {Path} -- [description]
        in_warp {Path} -- [description]
    """
    invwarp = fsl.InvWarp()
    invwarp.inputs.warp = in_warp
    invwarp.inputs.reference = ref
    invwarp.inputs.inverse_warp = out_warp
    return invert_warp


def apply_warp(in_file: Path, ref: Path, warp: Path, out_file: Path):
    """
    Apply non-linear warp file in an image to resample it to another one
    Arguments:
        in_file {Path} -- [Image to apply non-linear warp on]
        ref {Path} -- [Image to resample to]
        warp {Path} -- [non-linear warp file]
        out_file {Path} -- [Output image]
    """
    aw = fsl.ApplyWarp()
    aw.inputs.in_file = in_file
    aw.inputs.ref_file = ref
    aw.inputs.field_file = warp
    aw.inputs.out_file = out_file
    return aw


def apply_affine(in_file: Path, ref: Path, aff: Path, out_file: Path):
    """
    apply affine file to preform linear registration from one image to another
    Arguments:
        ref {Path} -- [reference image]
        aff {Path} -- [affine matrix file]
        out_file {Path} -- [output file]
        in_file {Path} -- [file to apply affine matrix on]
    """
    ax = fsl.ApplyXFM()
    ax.inputs.in_file = in_file
    ax.inputs.in_matrix_file = aff
    ax.inputs.out_file = out_file
    ax.inputs.reference = ref
    return ax


def highres2dwi(highres: Path, dwi: Path, out_file: Path, out_matrix_file: Path):
    """
    Perform linear registration from atlas in highres subjects` space to dwi` space
    Arguments:
        highres {Path} -- [atlas in subjects' highres space]
        dwi {Path} -- [subjects' dwi file]
        out_file {Path} -- [output linear registered file]
    """
    flt = fsl.FLIRT()
    flt.inputs.in_file = highres
    flt.inputs.reference = dwi
    flt.inputs.out_file = out_file
    flt.inputs.out_matrix_file = out_matrix_file
    return flt


def RoundAtlas(atlas_file: str):
    img = nib.load(atlas_file)
    orig_data = img.get_fdata()
    new_data = np.round(orig_data)
    new_img = nib.Nifti1Image(new_data.astype(int), img.affine)
    nib.save(new_img, atlas_file)

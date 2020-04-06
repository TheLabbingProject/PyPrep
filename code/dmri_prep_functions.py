import glob
import nipype.interfaces.mrtrix3 as mrt
import nipype.interfaces.fsl as fsl
import nibabel as nib
import os
import pathlib
from pathlib import Path
from nilearn.image import index_img


def GenIndex(epi_file: Path, index_file: Path):
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


def eddy_correct(
    in_file: Path,
    mask: Path,
    index: Path,
    acq: Path,
    bvec: Path,
    bval: Path,
    fieldcoef: Path,
    movpar: Path,
    out_base: Path,
):
    """
    Generating FSL's eddy-currents correction tool, eddy, with specific inputs.
    Arguments:
        in_file {Path} -- [Path to dwi file]
        mask {Path} -- [Path to brain mask file]
        index {Path} -- [Path to index.txt file]
        acq {Path} -- [Path to datain.txt file]
        bvec {Path} -- [Path to bvec file (extracted automatically when converting the dicom file)]
        bval {Path} -- [Path to bval file (extracted automatically when converting the dicom file)]
        fieldcoef {Path} -- [Path to field coeffient as extracted after topup procedure]
        movpar {Path} -- [Path to moving parameters as extracted after topup procedure]
        out_base {Path} -- [Path to eddy's output base name]

    Returns:
        eddy [type] -- [nipype's FSL eddy's interface, generated with specific inputs]
    """
    eddy = fsl.Eddy()
    eddy.inputs.in_file = in_file
    eddy.inputs.in_mask = mask
    eddy.inputs.in_index = index
    eddy.inputs.in_acqp = acq
    eddy.inputs.in_bvec = bvec
    eddy.inputs.in_bval = bval
    eddy.inputs.in_topup_fieldcoef = fieldcoef
    eddy.inputs.in_topup_movpar = movpar
    eddy.inputs.out_base = str(out_base)
    return eddy


# Initiate Mrtrix folder
def init_mrt_folder(folder_name: str, data_type: str):
    mrt_folder = folder_name.replace("Niftis", f"Derivatives/{data_type}_prep")
    if not os.path.isdir(mrt_folder):
        os.makedirs(mrt_folder)
    print(f"Generated temporary directory: {mrt_folder}")
    return mrt_folder


def Gen_dwi_mask(dwi_file: str, frac: float):
    AP_b0 = index_img(dwi_file, 0)
    b0_file = dwi_file.replace(".nii.gz", "_b0.nii.gz")
    nib.save(AP_b0, b0_file)

    bet = fsl.BET()
    bet.inputs.in_file = b0_file
    bet.inputs.mask = True
    bet.inputs.frac = frac
    bet.inputs.out_file = dwi_file.replace(".nii.gz", "_brain.nii.gz")
    print("Creating Brain mask...")
    print(bet.cmdline)
    bet.run()
    mask_file = dwi_file.replace(".nii.gz", "_brain_mask.nii.gz")
    os.remove(bet.inputs.out_file)
    os.remove(b0_file)
    return mask_file


# Convert DWI


def load_initial_files(folder_name: str, data_type: str):
    sub = folder_name.split(os.sep)[-1]
    dwi_folder = f"{folder_name}/dwi"
    anat_folder = f"{folder_name}/anat"
    fmap_folder = f"{folder_name}/fmap"
    func_folder = f"{folder_name}/func"
    for file in os.listdir(dwi_folder):
        if "dwi" in file:
            if "dwi.nii.gz" in file:
                dwi_file = f"{dwi_folder}/{file}"
            elif "bvec" in file:
                bvec = f"{dwi_folder}/{file}"
            elif "bval" in file:
                bval = f"{dwi_folder}/{file}"
    for file in os.listdir(func_folder):
        if "sbref.nii" in file:
            sbref_file = f"{func_folder}/{file}"
        elif "bold.nii" in file:
            func_file = f"{func_folder}/{file}"
    anat_file = f"{anat_folder}/{sub}_T1w.nii.gz"
    PA_file = glob.glob(f"{fmap_folder}/*.nii.gz")[0]
    if "dwi" in data_type:
        return anat_file, dwi_file, bvec, bval, PA_file
    elif "func" in data_type:
        return anat_file, func_file, sbref_file, dwi_file, PA_file


def convert_to_mif(in_file, out_file, bvec=None, bval=None):
    mrconvert = mrt.MRConvert()
    mrconvert.inputs.in_file = in_file
    mrconvert.inputs.out_file = out_file
    if bvec and bval:
        mrconvert.inputs.args = (
            f"-json_import {mrconvert.inputs.in_file.replace('nii.gz','json')}"
        )
        mrconvert.inputs.grad_fsl = (bvec, bval)
    mrconvert.run()
    return out_file


def Denoise(in_file: Path, in_mask: Path, out_file: Path):
    denoise = mrt.DWIDenoise()
    denoise.inputs.in_file = in_file
    denoise.inputs.noise = in_file.with_name(in_file.stem + "_noise.mif")
    denoise.inputs.mask = in_mask
    denoise.inputs.out_file = out_file
    print(denoise.cmdline)
    return denoise


def Unring(in_file: Path, out_file: Path):
    unring = mrt.MRDeGibbs()
    unring.inputs.in_file = in_file
    unring.inputs.out_file = out_file
    print(unring.cmdline)
    return unring


def DWI_prep(
    degibbs: Path, PA: Path, out_file: Path, data_type: str = "dwi", func=None
):
    eddy_dir = Path(degibbs.parent / "eddyqc")
    if not eddy_dir.is_dir():
        print("Creating directory for eddy current correction parameters...")
    ### extract first AP volume
    AP_b0 = Path(degibbs.parent / "AP_b0.mif")
    if not AP_b0.is_file():
        mrconvert = mrt.MRConvert()
        mrconvert.inputs.in_file = degibbs
        mrconvert.inputs.out_file = AP_b0
        mrconvert.inputs.coord = [3, 0]
        print(mrconvert.cmdline)
        mrconvert.run()
    ### concatenate AP and PA
    dual_phased = Path(degibbs.parent / "b0s.mif")
    if not dual_phased.is_file():
        cmd = (
            f"mrcat {AP_b0.absolute()} {PA.absolute()} {dual_phased.absolute()} -axis 3"
        )
        os.system(cmd)
    ### initiate dwipreproc (eddy)
    args = "-rpe_pair"
    eddyqc_text = "eddyqc/"
    cmd = f"dwipreproc {degibbs.absolute()} {out_file.absolute()} -pe_dir AP {args} -se_epi {dual_phased.absolute()} -eddyqc_text {degibbs.parent}/{eddyqc_text}"
    print(cmd)
    return cmd


def bias_correct(in_file: str, out_file: str):
    bs_correction = mrt.DWIBiasCorrect()
    bs_correction.inputs.in_file = in_file
    bs_correction.inputs.use_fsl = True
    bs_correction.inputs.out_file = out_file
    print(bs_correction.cmdline)
    bs_correction.run()
    return out_file


def convert_tmp_anat(in_file: str):
    if "mif" in in_file:
        out_file = f"{os.path.dirname(in_file)}/T1.nii"
    else:
        out_file = in_file.replace("nii.gz", "mif")
    if not os.path.isfile(out_file):
        mrconvert = mrt.MRConvert()
        mrconvert.inputs.in_file = in_file
        mrconvert.inputs.out_file = out_file
        mrconvert.inputs.args = "-strides +1,+2,+3"
        print(mrconvert.cmdline)
        mrconvert.run()
    return out_file


def T1_correction(in_file: str):
    in_file_nii = convert_tmp_anat(in_file)
    cmd = f"fsl_anat -i {in_file_nii} --noseg --nosubcortseg"
    print(cmd)
    os.system(cmd)
    os.remove(in_file_nii)
    bias_corr_brain = f"{os.path.dirname(in_file)}/T1.anat/T1_biascorr_brain.nii.gz"
    bias_corr_mask = f"{os.path.dirname(in_file)}/T1.anat/T1_biascorr_brain_mask.nii.gz"
    bias_corr_brain, bias_corr_mask = [
        convert_tmp_anat(corr_file) for corr_file in [bias_corr_brain, bias_corr_mask]
    ]
    return bias_corr_brain, bias_corr_mask


def fit_tensors(dwi_file: str, mask_file: str, dti_file: str):
    dilated_mask = mask_file.replace(".mif", "_dilated.mif")
    cmd = f"maskfilter {mask_file} dilate {dilated_mask} -npass 3"
    os.system(cmd)
    tsr = mrt.FitTensor()
    tsr.inputs.in_file = dwi_file
    tsr.inputs.in_mask = dilated_mask
    tsr.inputs.out_file = dti_file
    print(tsr.cmdline)
    tsr.run()
    comp = mrt.TensorMetrics()
    comp.inputs.in_file = dti_file
    comp.inputs.out_fa = dti_file.replace("dti", "fa")
    comp.inputs.args = "-force"
    print(comp.cmdline)
    comp.run()
    return comp.inputs.out_fa
    # tsr.inputs.args = '- | tensor2metrtic - -fa'


def gen_response(dwi_file: str, dwi_mask: str, working_dir: str):
    resp = mrt.ResponseSD()
    resp.inputs.in_file = dwi_file
    resp.inputs.algorithm = "dhollander"
    resp.inputs.csf_file = f"{working_dir}/response_csf.txt"
    resp.inputs.wm_file = f"{working_dir}/response_wm.txt"
    resp.inputs.gm_file = f"{working_dir}/response_gm.txt"
    resp.inputs.in_mask = dwi_mask
    print(resp.cmdline)
    resp.run()
    return resp.inputs.wm_file, resp.inputs.gm_file, resp.inputs.csf_file


def calc_fibre_orientation(
    dwi_file: str, dwi_mask: str, wm_resp: str, gm_resp: str, csf_resp: str
):
    fod_wm, fod_gm, fod_csf = [
        resp.replace("response", "FOD") for resp in [wm_resp, gm_resp, csf_resp]
    ]
    fod_wm, fod_gm, fod_csf = [
        odf.replace("txt", "mif") for odf in [fod_wm, fod_gm, fod_csf]
    ]
    fod = mrt.EstimateFOD()
    fod.inputs.algorithm = "msmt_csd"
    fod.inputs.in_file = dwi_file
    fod.inputs.wm_txt = wm_resp
    fod.inputs.wm_odf = fod_wm
    fod.inputs.gm_txt = gm_resp
    fod.inputs.gm_odf = fod_gm
    fod.inputs.csf_txt = csf_resp
    fod.inputs.csf_odf = fod_csf
    fod.inputs.mask_file = dwi_mask
    fod.inputs.max_sh = 10, 0, 0
    print(fod.cmdline)
    fod.run()
    return fod_wm, fod_gm, fod_csf


def gen_tissues_orient(fod_wm, fod_gm, fod_csf):
    tissues = f"{os.path.dirname(fod_wm)}/tissues.mif"
    cmd = f"mrconvert {fod_wm} - -coord 3 0 | mrcat {fod_csf} {fod_gm} - {tissues} -axis 3"
    print(cmd)
    os.system(cmd)
    return tissues


def DWI_to_T1_cont(dwi_file: str, dwi_mask: str, t1_file: str, t1_mask: str):
    meanbzero = dwi_file.replace(".mif", "_meanbzero.mif")
    cmd = f"dwiextract {dwi_file} -bzero - | mrcalc - 0.0 -max - | mrmath - mean -axis 3 {meanbzero}"
    print(cmd)
    os.system(cmd)
    dwi_pseudoT1 = dwi_file.replace(".mif", "_pseudoT1.mif")
    cmd = f"mrcalc 1 {meanbzero} -div {dwi_mask} -mult - | mrhistmatch nonlinear - {t1_file} {dwi_pseudoT1} -mask_input {dwi_mask} -mask_target {t1_mask}"
    print(cmd)
    os.system(cmd)
    T1_pseudobzero = f"{os.path.dirname(dwi_file)}/T1_pseudobzero.mif"
    cmd = f"mrcalc 1 {t1_file} -div {t1_mask} -mult - | mrhistmatch nonlinear - {meanbzero} {T1_pseudobzero} -mask_input {t1_mask} -mask_target {dwi_mask}"
    print(cmd)
    os.system(cmd)
    return meanbzero, dwi_pseudoT1, T1_pseudobzero


def reg_dwi_T1(
    dwi_file: str,
    t1_brain: str,
    dwi_pseudoT1: str,
    T1_pseudobzero: str,
    meanbzero: str,
    t1_mask: str,
    dwi_mask: str,
    run: bool,
):
    working_dir = os.path.dirname(dwi_file)
    t1_registered = f"{working_dir}/T1_registered.mif"
    t1_mask_registered = f"{working_dir}/T1_mask_registered.mif"
    if run:
        rig_t1_to_pseudoT1 = f"{working_dir}/rigid_T1_to_pseudoT1.txt"
        rig_t1_to_dwi = f"{working_dir}/rigid_T1_to_dwi.txt"
        rig_pseudob0_to_b0 = f"{working_dir}/rigid_pseudobzero_to_bzero.txt"
        cmd_1 = f"mrregister {t1_brain} {dwi_pseudoT1} -type rigid -mask1 {t1_mask} -mask2 {dwi_mask} -rigid {rig_t1_to_pseudoT1}"
        cmd_2 = f"mrregister {T1_pseudobzero} {meanbzero} -type rigid -mask1 {t1_mask} -mask2 {dwi_mask} -rigid {rig_pseudob0_to_b0}"
        cmd_3 = f"transformcalc {rig_t1_to_pseudoT1} {rig_pseudob0_to_b0} average {rig_t1_to_dwi}"
        cmd_4 = f"mrtransform {t1_brain} {t1_registered} -linear {rig_t1_to_dwi}"
        cmd_5 = f"mrtransform {t1_mask} {t1_mask_registered} -linear {rig_t1_to_dwi} -template {t1_registered} -interp nearest -datatype bit"
        for cmd in [cmd_1, cmd_2, cmd_3, cmd_4, cmd_5]:
            print(cmd)
            os.system(cmd)
    return t1_registered, t1_mask_registered


def five_tissue(t1_registered: str, t1_mask_registered):
    out_file = f"{os.path.dirname(t1_registered)}/5TT.mif"
    out_vis = f"{os.path.dirname(t1_registered)}/vis.mif"
    seg = mrt.Generate5tt()
    seg.inputs.in_file = t1_registered
    seg.inputs.out_file = out_file
    seg.inputs.algorithm = "fsl"
    cmd = seg.cmdline + f" -mask {t1_mask_registered}"
    print(cmd)
    os.system(cmd)
    os.system(f"5tt2vis {out_file} {out_vis}")
    return out_vis, out_file


def generate_tracks(fod_wm: str, seg_5tt: str):
    working_dir = os.path.dirname(fod_wm)
    tk = mrt.Tractography()
    tk.inputs.in_file = fod_wm
    tk.inputs.out_file = f"{working_dir}/tractogram.tck"
    tk.inputs.act_file = seg_5tt
    tk.inputs.backtrack = True
    tk.inputs.algorithm = "iFOD2"
    tk.inputs.max_length = 250
    tk.inputs.out_seeds = f"{working_dir}/seeds.nii"
    # tk.inputs.power = 3
    tk.inputs.crop_at_gmwmi = True
    tk.inputs.seed_dynamic = fod_wm
    tk.inputs.select = 350000
    print(tk.cmdline)
    tk.run()
    return tk.inputs.out_file


def convert_tck_to_trk(tracts_tck: str, dwi_file: str):
    from nibabel.streamlines import Field
    from nibabel.orientations import aff2axcodes

    dwi_file = convert_to_mif(dwi_file, dwi_file.replace("mif", "nii"))
    out_file = tracts_tck.replace("tck", "trk")
    nii = nib.load(dwi_file)
    header = {}
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))
    tck = nib.streamlines.load(tracts_tck)
    nib.streamlines.save(tck.tractogram, out_file, header=header)
    os.remove(dwi_file)
    return out_file

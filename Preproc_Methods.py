import dipy.reconst.dti as dti
import nibabel as nib
import numpy as np
import os, glob
from dipy.data import gradient_table
from nipype.interfaces import fsl, mrtrix3 as mrt
import dipy.denoise.noise_estimate as ne
from nilearn.image import resample_to_img
import shutil

CAT_BATCH = r"/home/gal/Brain_Networks/Derivatives/Scripts/cat12_template.m"
FSF_TEMP = r"/home/gal/Brain_Networks/Derivatives/Scripts/feat_preproc_template.fsf"


def load_initial_files(mother_dir: str, sub: str):
    """
    Initiate relevant files for preprocessing procedures. Note that {mother_dir} must be a BIDS compatible directory.
    Arguments:
        mother_dir {str} -- [Path to a BIDS compliant directory (should contain "mother_dir/sub-xx")]
        sub {str} -- ['sub-xx']

    Returns:
        [type] -- [anat, func, sbref, dwi, bvec, bval, phasediff files from subject's directory.]
    """
    folder_name = f"{mother_dir}/{sub}"
    dwi_folder = f"{folder_name}/dwi"
    anat_folder = f"{folder_name}/anat"
    fmap_folder = f"{folder_name}/fmap"
    func_folder = f"{folder_name}/func"
    anat_file = func_file = sbref_file = dwi_file = bvec = bval = phasediff = None
    for file in os.listdir(dwi_folder):
        if "dwi" in file:
            if "dwi.nii" in file:
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
    for file in os.listdir(anat_folder):
        if "T1w.nii" in file:
            anat_file = f"{anat_folder}/{file}"
    for file in os.listdir(fmap_folder):
        if "PA" in file:
            phasediff = f"{fmap_folder}/{file}"

    return anat_file, func_file, sbref_file, dwi_file, bvec, bval, phasediff


class run_BET:
    """
    in_file: string representing location for whole-head mri scan to be brain-extracted.
    *Done with FSL's BET
    """

    def __init__(self, in_file: str, out_file: str = None):
        self.in_file = in_file
        img = nib.load(in_file)
        f_name = os.path.splitext(os.path.splitext(in_file)[0])[0]
        if not out_file:
            self.out_file = f"{f_name}_brain.nii"
            self.mask_file = f"{f_name}_brain_mask.nii"
        else:
            self.out_file = out_file
            self.mask_file = (
                f"{os.path.splitext(os.path.splitext(out_file)[0])[0]}_mask.nii"
            )
        self.functional = img.ndim > 3

    def init_params(self, in_file: str, functional: bool):
        BET = fsl.BET()
        BET.inputs.in_file = in_file
        if functional:
            BET.inputs.functional = True
        else:
            BET.inputs.robust = True
        BET.inputs.mask = True
        BET.inputs.out_file = self.out_file
        BET.mask_file = self.mask_file
        return BET

    def run(self):
        BET = self.init_params(in_file=self.in_file, functional=self.functional)
        BET.run()
        return BET


### ADD CAT12 METHOD ###
class cat12:
    def __init__(self, in_file: str):
        """

        @param in_file: String representing location for whole-head mri scan to be brain-extracted.
        *Done with SPM's CAT12
        """
        import matlab.engine

        self.eng = matlab.engine.start_matlab()
        self.in_file = in_file
        self.script = f"{os.path.dirname(in_file)}/cat12_script.m"

    def decompress_file(self, in_file: str):
        """

        @param in_file: String representing location for whole-head mri scan to be brain-extracted.
        *Done with SPM's CAT12
        @return:
        """
        cmd = f"gunzip -k {in_file}"
        os.system(cmd)
        decompressed = in_file.replace(".gz", "")
        return decompressed

    def init_script(self, in_file: str, script: str):
        replacements = {"DATA_PATH": in_file}
        with open(CAT_BATCH) as template:
            with open(script, "w") as outfile:
                for line in template:
                    for src, target in replacements.items():
                        line = line.replace(src, target)
                    outfile.write(line)

    def run(self):
        decompressed = self.decompress_file(in_file=self.in_file)
        self.init_script(in_file=decompressed, script=self.script)
        self.eng.run(self.script, nargout=0)
        os.remove(decompressed)


class MotionCorrection:
    def __init__(self, in_file: str, out_file: str):
        """
        Motion correction with FSL's MCFLIRT
        @param in_file: path of 4D image
        """
        self.in_file = in_file
        self.out_file = out_file

    def init_params(self, in_file: str, out_file: str):
        mcflt = fsl.MCFLIRT()
        mcflt.inputs.in_file = in_file
        mcflt.inputs.out_file = out_file
        # mcflt.inputs.save_mats = True
        # mcflt.inputs.save_plots = True
        # mcflt.inputs.save_rms = True
        # mcflt.inputs.stats_imgs = True
        return mcflt

    def run(self):
        mcflt = self.init_params(in_file=self.in_file, out_file=self.out_file)
        mcflt.run()
        return mcflt


class TopUp:
    def __init__(self, func_file: str, sbref: str, tmp_dir: str):
        self.func_file = func_file
        self.sbref = sbref
        self.tmp_dir = tmp_dir

    def run(self):
        tmp_dir = self.tmp_dir
        img = nib.load(self.func_file)
        func_new_header = self.func_file.split(os.sep)[-1].replace(
            ".nii.gz", "_b0.nii.gz"
        )
        func_new_header = f"{tmp_dir}/{func_new_header}"
        func_b0 = nib.Nifti1Image(img.get_fdata()[:, :, :, 0], img.affine)
        nib.save(func_b0, func_new_header)
        cmd = (
            f"fslmerge -t {tmp_dir}/Dual_phase_b0.nii.gz {func_new_header} {self.sbref}"
        )
        print(cmd)
        os.system(cmd)
        acq = f"{tmp_dir}/acqparams.txt"
        cmd = r'printf "0 1 0 0.072\n0 -1 0 0.072" > ' + acq
        print(cmd)
        os.system(cmd)
        idx = "1 " * img.shape[3]
        idx_file = open(f"{tmp_dir}/index.txt", "w")
        idx_file.write(idx)
        idx_file.close()
        topup_res, topup_field = self.TopUp(
            AP_PA=f"{tmp_dir}/Dual_phase_b0.nii.gz", acq=acq
        )
        idx_path = f"{tmp_dir}/index.txt"
        return topup_res, topup_field, idx_path, acq

    def TopUp(self, AP_PA: str, acq: str):
        print("Going through TOPUP procedure...")
        topup = fsl.TOPUP()
        topup.inputs.in_file = AP_PA
        topup.inputs.encoding_file = acq
        topup.inputs.out_base = f"{os.path.dirname(AP_PA)}/topup_res"
        topup.inputs.out_corrected = f"{os.path.dirname(AP_PA)}/topup_nifti_res.nii.gz"
        print(topup.cmdline)
        topup.run()
        return topup.inputs.out_corrected, f"{topup.inputs.out_base}_fieldcoef.nii.gz"


class PrepareEddy:
    def __init__(self, subj_dir: str, motion_corrected: str, phasediff: str):
        """
        Prepare data for eddy current correction.
        Includes mask creation for {motion_corrected} and TopUp procedure.
        Arguments:
            subj_dir {str} -- [path to subject's derivatives directory]
            motion_corrected {str} -- [path to motion corrected 4D image's file]
            phasediff {str} -- [path to other-phase image (PA)]
        """
        # self.AP = glob.glob(f"{subj_dir}/dwi/*_motion_corrected.nii.gz")[0]
        self.AP = motion_corrected
        # self.PA = glob.glob(f"{subj_dir}/dwi/*PA*.nii.gz")[0]
        self.PA = phasediff
        self.subj_dir = subj_dir

    def run(self):
        subj_dir = self.subj_dir
        AP = self.AP
        PA = self.PA
        img = nib.load(AP)
        AP_new_header = AP.replace(".nii.gz", "_b0.nii.gz")
        AP_new_header = f"{subj_dir}/dwi/tmp/{AP_new_header.split(os.sep)[-1]}"
        AP_b0 = nib.Nifti1Image(img.get_fdata()[:, :, :, 0], img.affine)
        nib.save(AP_b0, AP_new_header)
        cmd = f"fslmerge -t {subj_dir}/dwi/tmp/AP_PA_b0.nii.gz {AP_new_header} {PA}"
        print(cmd)
        os.system(cmd)
        acq = f"{subj_dir}/dwi/tmp/acqparams.txt"
        cmd = r'printf "0 -1 0 0.072\n0 1 0 0.072" > ' + acq
        print(cmd)
        os.system(cmd)
        idx = "1 " * img.shape[3]
        idx_file = open(f"{subj_dir}/dwi/tmp/index.txt", "w")
        idx_file.write(idx)
        idx_file.close()
        topup_res = self.TopUp(AP_PA=f"{subj_dir}/dwi/tmp/AP_PA_b0.nii.gz", acq=acq)
        brain_mask = self.CreateMask(AP_new_header)
        idx_path = f"{subj_dir}/dwi/tmp/index.txt"
        return topup_res, brain_mask, idx_path, acq

    def CreateMask(self, in_file):
        print("Creating brain mask...")
        BET = fsl.BET()
        BET.inputs.in_file = in_file
        BET.inputs.robust = True
        BET.inputs.mask = True
        BET.inputs.out_file = in_file.replace(".nii.gz", "_brain.nii.gz")
        BET.mask_file = in_file.replace(".nii.gz", "_brain_mask.nii.gz")
        print(BET.cmdline)
        BET.run()
        return BET.mask_file

    def TopUp(self, AP_PA: str, acq: str):
        print("Going through TOPUP procedure...")
        topup = fsl.TOPUP()
        topup.inputs.in_file = AP_PA
        topup.inputs.encoding_file = acq
        topup.inputs.out_base = f"{os.path.dirname(AP_PA)}/topup_res"
        topup.inputs.out_corrected = f"{os.path.dirname(AP_PA)}/topup_nifti_res.nii.gz"
        print(topup.cmdline)
        topup.run()
        return topup.inputs.out_base


class EddyCorrect:
    def __init__(self, in_file, fieldcoef, brain_mask, idx_file, bvec, bval, acq):
        print("Initializing EDDY correction...")
        self.in_file = in_file
        self.bval = bval
        self.bvec = bvec
        self.fieldcoef = fieldcoef
        self.brain_mask = brain_mask
        self.acq = acq
        self.idx_file = idx_file

    def init_params(self):
        eddyc = fsl.Eddy()
        eddyc.inputs.in_file = self.in_file
        eddyc.inputs.in_bval = self.bval
        eddyc.inputs.in_bvec = self.bvec
        eddyc.inputs.in_index = self.idx_file
        eddyc.inputs.in_acqp = self.acq
        eddyc.inputs.in_mask = self.brain_mask
        eddyc.inputs.in_topup_fieldcoef = self.fieldcoef
        eddyc.inputs.in_topup_movpar = self.fieldcoef.replace(
            "fieldcoef.nii.gz", "movpar.txt"
        )
        eddyc.inputs.out_base = os.path.join(
            os.path.dirname(self.in_file), "tmp", "diff_corrected"
        )
        # eddyc.inputs.out_corrected = os.path.join(os.path.dirname(self.in_file),'diff_corrected.nii.gz')
        print(eddyc.cmdline)
        return eddyc

    def run(self):
        eddyc = self.init_params()
        eddyc.run()


class mag_brain_extract:
    def __init__(self, field_mag: str):
        self.field_mag = field_mag

    def set_bet(self):
        bet = fsl.BET()
        bet.inputs.in_file = self.field_mag
        bet.inputs.out_file = (
            f"{os.path.dirname(self.field_mag)}/field_mag_brain.nii.gz"
        )
        return bet

    def run(self):
        bet = self.set_bet()
        bet.run()
        return bet.inputs.out_file


class GenerateWhiteMatterMask:
    def __init__(self, sub_dir: str):
        self.sub_dir = sub_dir
        self.anat_dir = f"{self.sub_dir}/anat"
        self.dwi_dir = f"{self.sub_dir}/dwi"

    def gather_files(self):
        anat_seg = glob.glob(f"{self.anat_dir}/mri/p0*")[0]
        target_file = glob.glob(f"{self.dwi_dir}/diff_corrected.nii.gz")[0]
        return anat_seg, target_file

    def resample_seg(self, target: str, seg: str):
        new_seg = resample_to_img(seg, target)
        nib.save(
            new_seg, os.path.join(os.path.dirname(target), "tmp", "Segmentation.nii.gz")
        )
        return new_seg

    def run(self):
        seg, target = self.gather_files()
        new_seg = self.resample_seg(target, seg)


class DenoiseDwi:
    def __init__(self, in_file: str):
        """

        @param in_file: String representing 4D dwi image that underwent brain extraction (needed to use the _brain_mask image for denoising).
        Creates denoised 4D dwi image.
        """
        self.dwi_file = in_file
        self.mask = in_file.replace(".nii.gz", "_brain_mask.nii.gz")
        self.out_file = in_file.replace(".nii.gz", "_denoised.nii.gz")
        self.out_noise = in_file.replace(".nii.gz", "noise.nii.gz")

    def init_params(self, dwi_file: str, mask: str, out_file: str, out_noise: str):
        denoise = mrt.DWIDenoise()
        denoise.inputs.in_file = dwi_file
        denoise.inputs.mask = mask
        denoise.inputs.out_file = out_file
        denoise.inputs.noise = out_noise
        return denoise

    def run(self):
        denoise = self.init_params(
            dwi_file=self.dwi_file,
            mask=self.mask,
            out_file=self.out_file,
            out_noise=self.out_noise,
        )
        denoise.run()


class Reconstract_tensors:
    def __init__(self, in_file: str):
        """

        @param in_file: String representing initial 4-D dwi image that underwent brain extraction and denoising.
        Creates FA,MD,RGB images by fitting a tensor model relying on bvec and bvals.
        """
        self.denoised = in_file.replace(".nii.gz", "_denoised.nii.gz")
        self.masked_data = nib.load(self.denoised).get_fdata()
        bvec_file = in_file.replace(".nii.gz", ".bvec")
        bval_file = in_file.replace(".nii.gz", ".bval")
        self.img = nib.load(in_file)
        self.gtab = gradient_table(bval_file, bvec_file)

    def fit_tensor(self, gtab, masked_data):
        print(
            f"Fitting Tensor Model for {self.denoised.split(os.sep)[-1]}.\nPlease wait."
        )
        # sigma = ne.estimate_sigma(masked_data)
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(masked_data)
        return tenfit

    def compute_anisotropy(self, tenfit: dti.TensorFit):
        print("Computing anisotropy measures (FA, MD, RGB)")
        mother_dir = os.path.dirname(self.denoised)
        FA = dti.fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0
        fa_img = nib.Nifti1Image(FA.astype(np.float32), self.img.affine)
        nib.save(fa_img, f"{mother_dir}/tensor_fa.nii.gz")
        evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), self.img.affine)
        nib.save(evecs_img, f"{mother_dir}/tensor_evecs.nii.gz")
        MD1 = dti.mean_diffusivity(tenfit.evals)
        MD1_img = nib.Nifti1Image(MD1.astype(np.float32), self.img.affine)
        nib.save(MD1_img, f"{mother_dir}/tensors_md.nii.gz")
        FA = np.clip(FA, 0, 1)
        RGB = dti.color_fa(FA, tenfit.evecs)
        RGB_img = nib.Nifti1Image(np.array(255 * RGB, "uint8"), self.img.affine)
        nib.save(RGB_img, f"{mother_dir}/tensor_rgb.nii.gz")

    def run(self):
        tenfit = self.fit_tensor(gtab=self.gtab, masked_data=self.masked_data)
        self.compute_anisotropy(tenfit=tenfit)


class Create_Atlas:
    def __init__(self, atlas_dir: str, target_dir: str):
        self.atlas_dir = atlas_dir
        self.target_dir = target_dir

    def get_paths(self):
        target_dir = self.target_dir
        atlas_dir = self.atlas_dir
        dir_split = target_dir.split(os.sep)
        subj = dir_split[-1]
        print(subj)
        proj_dir = os.path.join("/", dir_split[1], dir_split[2], dir_split[3])
        highres = f"{target_dir}/anat/{subj}_T1w_brain.nii.gz"
        lowres = f"{target_dir}/dwi/diff_corrected.nii.gz"
        out_dir = f"{proj_dir}/Derivatives/Registrations/{subj}/Atlases_and_Transforms"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        atlas_highres = f"{atlas_dir}/MegaAtlas_Labels_highres.nii"
        atlas_labels = f"{atlas_dir}/MegaAtlas_cortex_Labels.nii"
        return highres, lowres, out_dir, atlas_highres, atlas_labels

    def FLIRT(self, in_file: str, ref: str, out_dir: str, type: int):
        """
        FLIRT linear transformation
        @param in_file: image to be transformed
        @param ref: image to transform into
        @param out_dir: directory for outputs
        @param type: 0 for highres2lowres, 1 for atlasHighres2highres, 2 for atlasLabels2highres
        @return:
        """
        flt = fsl.FLIRT()
        flt.inputs.in_file = in_file
        flt.inputs.reference = ref
        flt.inputs.bins = 256
        flt.inputs.cost = "corratio"
        flt.inputs.searchr_x = [-90, 90]
        flt.inputs.searchr_y = [-90, 90]
        flt.inputs.searchr_z = [-90, 90]
        flt.inputs.dof = 12
        flt.inputs.interp = "trilinear"
        if type == 0:
            flt.inputs.out_file = f"{out_dir}/Highres2Lowres_linear.nii.gz"
            flt.inputs.out_matrix_file = f"{out_dir}/Highres2Lowres_linear.mat"
            print("Running linear transformation from Highres to Lowres:")
        elif type == 1:
            flt.inputs.out_file = f"{out_dir}/HighresAtlas2Highres_linear.nii.gz"
            flt.inputs.out_matrix_file = f"{out_dir}/HighresAtlas2Highres_linear.mat"
            print("Running linear transformation from Atlas to Highres:")
        elif type == 2:
            flt.inputs.out_file = f"{out_dir}/LabelsAtlas2Highres_linear.nii.gz"
            flt.inputs.out_matrix_file = f"{out_dir}/LabelsAtlas2Highres_linear.mat"
            print("Running linear transformation from Atlas to Highres:")
        print(flt.cmdline)
        flt.run()
        return flt.inputs.out_matrix_file

    def FNIRT(self, in_file: str, ref: str, out_dir: str, type: int, aff: str):
        """
        FNIRT non-linear transformation
        @param in_file: image to be transformed
        @param ref: image to transform into
        @param out_dir: directory for outputs
        @param type: 0 for highres2lowres, 1 for atlasHighres2highres, 2 for atlasLabels2highres
        @return:
        """
        fnt = fsl.FNIRT()
        fnt.inputs.in_file = in_file
        fnt.inputs.ref_file = ref
        fnt.inputs.affine_file = aff
        if type == 0:
            fnt.inputs.warped_file = f"{out_dir}/Highres2Lowres.nii.gz"
            print("Running non-linear transformation from Highres to Lowres:")
        elif type == 1:
            fnt.inputs.warped_file = f"{out_dir}/HighresAtlas2Highres.nii.gz"
            print("Running non-linear transformation from Atlas to Highres:")
        elif type == 2:
            fnt.inputs.warped_file = f"{out_dir}/LabelsAtlas2Highres.nii.gz"
            print("Running nonlinear transformation from Atlas to Highres:")
        print(fnt.cmdline)
        fnt.run()
        return fnt.inputs.warped_file

    def RoundAtlas(self, atlas_file: str):
        img = nib.load(atlas_file)
        orig_data = img.get_fdata()
        new_data = np.round(orig_data)
        new_img = nib.Nifti1Image(new_data.astype(int), img.affine)
        nib.save(new_img, atlas_file)

    def resample_atlas(self, atlas_file: str, lowers, aff):
        applyxfm = fsl.ApplyXFM()
        applyxfm.inputs.in_file = atlas_file
        applyxfm.inputs.reference = lowers
        applyxfm.inputs.in_matrix_file = aff
        applyxfm.inputs.out_file = atlas_file.replace(".nii.gz", "_resampled.nii.gz")
        applyxfm.inputs.apply_xfm = True
        print(applyxfm.cmdline)
        applyxfm.run()
        return applyxfm.inputs.out_file

    def run(self):
        highres, lowres, out_dir, atlas_highres, atlas_labels = self.get_paths()
        # HIGHRES - LOWRES
        highres2lowers_aff = f"{out_dir}/Highres2Lowres_linear.mat"
        warped_file = f"{out_dir}/Highres2Lowres.nii.gz"
        if not os.path.isfile(highres2lowers_aff):
            highres2lowers_aff = self.FLIRT(
                in_file=highres, ref=lowres, out_dir=out_dir, type=0
            )
        if not os.path.isfile(warped_file):
            warped_file = self.FNIRT(
                in_file=highres, ref=lowres, out_dir=out_dir, type=0, aff=aff
            )
        # ATLAS HIGHRES - HIGHRES
        aff = f"{out_dir}/HighresAtlas2Highres_linear.mat"
        warped_file = f"{out_dir}/HighresAtlas2Highres.nii.gz"
        if not os.path.isfile(aff):
            aff = self.FLIRT(
                in_file=atlas_highres, ref=highres, out_dir=out_dir, type=1
            )
        if not os.path.isfile(warped_file):
            warped_file = self.FNIRT(
                in_file=atlas_highres, ref=highres, out_dir=out_dir, type=1, aff=aff
            )
        self.RoundAtlas(warped_file)
        resampled = self.resample_atlas(warped_file, lowres, highres2lowers_aff)
        self.RoundAtlas(resampled)
        # ATLAS LABELS - HIGHRES
        aff = f"{out_dir}/LabelsAtlas2Highres_linear.mat"
        warped_file = f"{out_dir}/LabelsAtlas2Highres.nii.gz"
        if not os.path.isfile(aff):
            aff = self.FLIRT(in_file=atlas_labels, ref=highres, out_dir=out_dir, type=2)
        if not os.path.isfile(warped_file):
            warped_file = self.FNIRT(
                in_file=atlas_labels, ref=highres, out_dir=out_dir, type=2, aff=aff
            )
        self.RoundAtlas(warped_file)
        resampled = self.resample_atlas(warped_file, lowres, highres2lowers_aff)
        self.RoundAtlas(resampled)


class MakeSubjectAtlas:
    def __init__(self, sub_dir: str, atlas_dir: str):
        self.sub_dir = sub_dir
        self.atlas_dir = atlas_dir
        self.subj = sub_dir.split(os.sep)[-1]
        self.proj_name = sub_dir.split(os.sep)[-3]
        self.derivatives = f"/home/gal/{self.proj_name}/Derivatives"
        # self.fsf_dir = f'{self.derivatives}/Scripts/prep_fsfs'
        # if not os.path.isdir(self.fsf_dir):
        #     os.mkdir(self.fsf_dir)

    def Prep4Feat(self):
        anat_dir = f"{self.sub_dir}/anat"
        seg = glob.glob(f"{anat_dir}/mri/p0*.nii")[0]
        new_seg = f"{anat_dir}/{self.subj}_T1w_brain.nii.gz"
        shutil.copy(seg, new_seg)
        dwi_img = glob.glob(f"{self.sub_dir}/dwi/diff_corrected.nii.gz")[0]
        img = nib.load(dwi_img)
        IMG_size = str(np.sum(img.get_fdata() > 0))
        TR = str(img.header.get_zooms()[3])
        ntime = str(img.shape[-1])
        fsf_loc = f"{self.fsf_dir}/{self.subj}_dwi.fsf"
        outdir = f"{self.derivatives}/Registrations/{self.subj}/Feat_reg.feat"
        return new_seg, dwi_img, IMG_size, TR, ntime, fsf_loc, outdir

    def GenReplacements(self, new_seg, dwi_img, IMG_size, TR, ntime, outdir):
        replacements = {
            "NTPTS": ntime,
            "OUT_DIR": outdir,
            "cur_TR": TR,
            "DWI_IMG": dwi_img,
            "MPRAGE": new_seg,
            "img_size": IMG_size,
        }
        return replacements

    def GenFsf(self, replacements, fsf_loc):
        fsf_template = FSF_TEMP
        new_fsf = fsf_loc
        with open(fsf_template) as infile:
            with open(new_fsf, "w") as out_file:
                for line in infile:
                    for src, target in replacements.items():
                        line = line.replace(src, target)
                    out_file.write(line)
        print(f"Created fsf for {self.subj} at {new_fsf}")

    def run_fsf(self, fsf: str):
        feat = fsl.model.FEAT()
        feat.inputs.fsf_file = fsf
        feat.run()

    def reg_atlas2subj(self, atlas_dir, feat_dir):
        highres_atlas = glob.glob(f"{atlas_dir}/*highres.nii")[0]
        labels_atlas = glob.glob(f"{atlas_dir}/*labels.nii")[0]
        affine = f"{feat_dir}/reg/standard2highres.mat"
        ref = f"{feat_dir}/reg/highres.nii.gz"
        for in_file in [highres_atlas, labels_atlas]:
            if "highres" in in_file:
                out_file = f"{self.derivatives}/Registrations/{self.subj}/Atlases_and_Transforms/HighresAtlas2Highres.nii.gz"
            elif "labels" in in_file:
                out_file = f"{self.derivatives}/Registrations/{self.subj}/Atlases_and_Transforms/LabelsAtlas2Highres.nii.gz"
            self.apply_affine(in_file, affine, out_file, ref)

    def apply_affine(self, in_file, affine, out_file, ref):
        xfm = fsl.ApplyXFM()
        xfm.inputs.in_file = in_file
        xfm.inputs.reference = ref
        xfm.inputs.in_matrix_file = affine
        xfm.inputs.apply_xfm = True
        xfm.inputs.out_file = out_file
        xfm.inputs.out_matrix_file = out_file.replace("nii.gz", "mat")
        print("Applyting linear transformation on atlas:")
        print(xfm.cmdline)
        xfm.run()


class apply_warp:
    def __init__(self, in_file: str, out_file: str, field_file: str, ref: str):
        self.in_file = in_file
        self.out_file = out_file
        self.field_file = field_file
        self.ref = ref

    def init_fsl(self, in_file: str, ref: str, out_file: str, field_file: str):
        aw = fsl.ApplyWarp()
        aw.inputs.in_file = in_file
        aw.inputs.out_file = out_file
        aw.inputs.field_file = field_file
        aw.inputs.ref_file = ref
        return aw

    def run(self):
        aw = self.init_fsl(self.in_file, self.ref, self.out_file, self.field_file)
        print(aw.cmdline)
        aw.run()


class apply_XFM:
    def __init__(self, in_file: str, ref: str, out_file: str):
        self.in_file = in_file
        self.ref = ref
        self.out_file = out_file
        self.out_mat = out_file.replace("nii.gz", "mat")

    def init_fsl(self, in_file: str, ref: str, out_file: str, out_mat: str):
        ax = fsl.FLIRT()
        ax.inputs.in_file = in_file
        ax.inputs.reference = ref
        ax.inputs.out_file = out_file
        ax.inputs.out_matrix_file = out_mat
        return ax

    def run(self):
        ax = self.init_fsl(self.in_file, self.ref, self.out_file, self.out_mat)
        print(ax.cmdline)
        ax.run()

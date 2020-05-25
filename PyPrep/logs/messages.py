from pathlib import Path

PRINT_START = """Initial ("raw") images found on {subj} directory:
    Subject's sub-directory in a BIDS-compatible directory: {subj_dir}
    - Structural image: {anat}
    - Functional image: {func}
    - Diffusion-weighted image (DWI): {dwi}
    - DWI's .bvec and .bval files: {bvec}, {bval}
    - Inverse (PA) encoded DWI: {phasediff}
"""
BIDSVALIDATOR = """ BIDS validator.
    Inputs:
        - Directory to be validated: {bids_dir}.
    Outputs:
        - Validation of BIDS format (and an error for an incompatible directory.)
    """

GENERATEOUTPUT = """Derivatives directory generator (for a BIDS-compatible one.)
    Inputs:
        - BIDS-compatible directory: {bids_dir}
    Outputs:
        - Derivatives directory that resembles the input: {derivatives_dir}
    """

GENERATEFIELDMAP = """Fieldmap generator.
    Working directory: {output_dir}
    Inputs:
        - AP-encoded DWI image: {AP}
        - PA-encoded (Inverse-encoded) DWI image: {PA}
    Outputs: (all under the working directory)
        - Dual-phased encoded image: {merged}
        - Data file (required for eddy-currents correction): {datain}
        - Index file (required for FSL's TopUp): {index}
        - Fieldmap image: {fieldmap}
        - Fieldmap image in radians (required for FSL's FEAT): {fieldmap_rad}
        - Fieldmap magnitude image (also required for FSL's FEAT): {fieldmap_mag}
        - Brain-extracted fieldmap magnitude image: {fieldmap_brain}
    """

BETBRAINEXTRACTION = """Brain extraction using FSL's BET.
    Inputs:
        - Whole-head image: {in_file}
        - Output, brain exracted image: {out_file}
    Outputs:
        - Brain-extracted image: {out_file}
        (note, if no brain-extracted image given as input, BET will automatically generate it at the same directory as whole-headed one, with "_brain" suffix.)
        """

FEAT = """FSL's FEAT analysis generator.
    Inputs:
        - Functional MR (main) image: {fmri}
        - High resolution (structural) image: {highres}
        - Brain-extracted fieldmap magnitude image: {fielmap_brain}
        - Fieldmap image (in radians): {fieldmap_rad}
    Outputs:
        - FEAT's .fsf design file: {subj_design}
        - FEAT output directory: {out_feat}
"""

INITIATEMRTRIX = """Mrtrix3's preprocessing directory generator.
    Mrtrix directory: {mrtrix_dir}
    Inputs:
        - DWI image: {dwi}
        - Structural image: {anat}
        - Dual-encoded (AP-PA) image: {phasediff}
    Outputs:
        Same files (under Mrtrix directory) in ".mif" format for compatability with Mrtrix3's tools.
"""

DENOISEDWI = """Mrtrix3's initial denoiser.
    Inputs:
        - Motion-corrected DWI image: {epi_file}
        - Brain mask image: {mask}
    Outputs:
        - Initially denoised DWI image: {denoised}
"""

UNRING = """ Mrtrix3's gibbs rings removal.
    Inputs:
        - Initially denoised DWI image: {denoised}
    Outputs:
        - Gibbs-removed DWI image: {degibbs}
"""

DWIPREPROCESS = """Mrtrix3's Eddy currents and various other geometric corrections tool.
    Inputs:
        - Gibbs-removed DWI image: {degibbs}
        - Dual-encoded (AP-PA) image: {phasediff}
    Outputs:
        - Preprocessed DWI image: {preprocessed}
"""

CONVERT2NIFTI = """Converter of Mrtrix3's .mif format files to the more commonly used nifti (.nii.gz) format.
    Inputs:
        - Mrtrix preprocessing directory (containing .mif files): {mrtrix_dir}
        - Subject's derivatives directory, containing "dwi", "anat", etc. sub-directories: {subj_dir}
    Outputs:
        - Nifti (.nii.gz) files, converted from Mrtrix preprocessing directory to their relevant sub-directories.
"""

BIASCORRECTION = """Initial B1 bias field correction using Mrtrix3's dwibiascorrect.
    Inputs:
        - Preprocessed (denoised, degibbs, eddy-corrected) DWI image: {preprocessed}
    Outputs:
        - Bias-corrected image: {bias_corrected}
"""

FIVETISSUES = """Generation of five-tissues-type (5TT) image, and Registration of DWI image to structual one (and vice versa).
    Working directory: {parent}
    Inputs:
        - DWI image: {dwi}
        - DWI mask: {dwi_mask}
        - Structual image: {anat}
        - Structual mask: {anat_mask}
    Outputs:
        - DWI mean B0 image: {meanbzero}
        - DWI image initially registered to structural one: {dwi_pseudo_t1}
        - Structural image initiall registered to DWI: {t1_pseudo_dwi}
        - Structual image fully registered to DWI: {t1_registered}
        - Registered structural image's mask: {t1_reg_mask}
        - Five-tissues-type (5TT) image: {five_tissue}
        - Visualisation-compatible form of 5TT image: {vis}
"""

PREPROCESSANAT = """Perform strutural preprocessing using FSL's fsl_anat script.
    Working directory: {init_anat}
    Inputs:
        - Raw structural image: {anat}
    Outputs:
        - Sub-directory contining structural preprocessing files: {anat_dir}
"""

# PyPrep
## Yaniv Assaf's lab automated preprocessing project.
### Prepocessing pipeline:
```
1. Generate fieldmap files for eddy current correction:
    A. Merge both phase-encoded images into one. (FSL's Merge)
    B. Create acuisition parameters file, relying on the dual-phase encoded image.
    C. Create index file (a file contining n-rows of 1 for each scan of the image to be preprocessed.)
    D. Apply TopUp to remove suceptabillity bias.
    E. Create fieldmap magnitude brain image and transform fieldmap into radians (for compatibillity with later used tools.)
2. Generate brain-extracted structural image and mask (FSL's BET)
3. For functional images, run FSL's FEAT using the functional, brain-extracted structural,brain extracted fieldmap magnitude and fieldmap (in radians) images.
4. For diffusion images, apply eddy-currents correction using the DWI, brain mask, index file, acuisition parameters file, bvec, bval, and field coefficients and movpar files from TopUp procedure.
```

### Output:
* derivatives/
    * sub-01/
        * dwi/
            * sub-01_acq-AP_dwi_MotionCorrected.nii
            * dwi.feat/
                * {FEAT output files}
                * reg/
                    * {FEAT registration files}
                mc/
                    * Motion correction output files
                    * prefiltered_func_data_mcf.mat/
        * fmap/
            * Fieldmap files
        * anat/
            * Subject's processed structural images
        * scripts/
            * Subject's scripts for FEAT
        * atlases/
            * Atlas's registration to subject's-space file
        * func/
            * sub-01_task-rest_bold_MotionCorrected.nii
            * func.feat/
                * {FEAT output files}
                * reg/
                    * {FEAT registration files}
                mc/
                    * Motion correction output files
                    * prefiltered_func_data_mcf.mat/
    * sub-02/
        * dwi/
        * fmap/
        * anat/
        * scripts/
        * atlases/
        * func/
    * .
    * .
    * .

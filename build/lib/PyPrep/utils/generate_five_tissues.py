from pathlib import Path
from logs import messages
from .utillities import (
    FSLOUTTYPE,
    five_tissue,
    reg_dwi_to_t1,
    dwi_to_T1_coreg,
    convert_to_mif,
)
from .file_utils import check_files_existence


class GenerateFiveTissue:
    def __init__(self, dwi: Path, dwi_mask: Path, anat: Path, anat_mask: Path):
        self.dwi = dwi
        self.dwi_mask = dwi_mask
        self.anat = anat
        self.anat_mask = anat_mask
        self.new_highres_mask = Path(anat.parent / f"{anat.stem}_brain_mask.mif")
        self.meanbzero = Path(dwi.parent / f"{dwi.stem}_meanbzero.mif")
        self.dwi_pseudoT1 = Path(dwi.parent / f"{dwi.stem}_pseudoT1.mif")
        self.T1_pseudobzero = Path(dwi.parent / f"{dwi.stem}_T1_pseudobzero.mif")
        self.t1_registered = Path(dwi.parent / "T1_registered.mif")
        self.t1_mask_registered = Path(dwi.parent / "T1_mask_registered.mif")
        self.five_tissues = self.t1_registered.parent / "5TT.mif"
        self.vis = self.t1_registered.parent / "vis.mif"
        self.exist_first_step = check_files_existence(
            [self.t1_registered, self.t1_mask_registered]
        )
        self.exist_second_step = check_files_existence([self.five_tissues, self.vis])

    def __str__(self):
        str_to_print = messages.FIVETISSUES.format(
            parent=self.dwi.parent,
            dwi=self.dwi,
            dwi_mask=self.dwi_mask,
            anat=self.anat,
            anat_mask=self.anat_mask,
            meanbzero=self.meanbzero,
            dwi_pseudo_t1=self.dwi_pseudoT1,
            t1_pseudo_dwi=self.T1_pseudobzero,
            t1_registered=self.t1_registered,
            t1_reg_mask=self.t1_mask_registered,
            five_tissue=self.five_tissues,
            vis=self.vis,
        )
        return str_to_print

    def register_dwi_to_t1(self):
        if not self.new_highres_mask.is_file():
            self.new_highres_mask = convert_to_mif(
                self.anat_mask, self.new_highres_mask
            )
        if not self.T1_pseudobzero.is_file():
            self.meanbzero, self.dwi_pseudoT1, self.T1_pseudobzero = dwi_to_T1_coreg(
                self.dwi, self.dwi_mask, self.anat, self.new_highres_mask
            )
        if not self.t1_registered.is_file():
            self.t1_registered, self.t1_mask_registered = reg_dwi_to_t1(
                self.dwi,
                self.anat,
                self.dwi_pseudoT1,
                self.T1_pseudobzero,
                self.meanbzero,
                self.anat_mask,
                self.dwi_mask,
                True,
            )

    def generate_five_tissues(self):
        self.vis, self.five_tissues = five_tissue(
            self.t1_registered, self.t1_mask_registered
        )

    def run(self):
        if not self.exist_first_step:
            print(
                "Performing registration of DWI image to structural one (and vice versa)."
            )
            self.register_dwi_to_t1()
        else:
            print(
                "Registration between DWI image and structural already done. Continuing."
            )
        if not self.exist_second_step:
            print(
                "Generating five-tissue-type (5TT) image (for later usage of the white-matter tissue mask)."
            )
            self.generate_five_tissues()
        else:
            print("Five-tissue-type (5TT) file already exists. Continuing.")

        return self.five_tissues, self.vis

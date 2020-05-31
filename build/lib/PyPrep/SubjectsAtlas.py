import glob
import os
import time
import tkinter as tk
from pathlib import Path
import PyPrep.registrations
from PyPrep.diffusion import dmri_prep_functions as dmri_methods
from PyPrep.functional import fmri_prep_functions as fmri_methods
from PyPrep.registrations import registrations_functions as reg_functions
from atlases.atlases import Atlases
from templates.templates import Templates
from tkinter import filedialog

ATLAS = Atlases.megaatlas.value
FSLOUTTYPE = ".nii.gz"


# class InvertWarp:
#     def __init__(self, in_file: Path, ref: Path, out_dir: Path, warp: Path):
#         self.in_file = in_file
#         self.ref = ref
#         self.out_dir = out_dir
#         self.warp = warp
#         self.out_warp = out_dir / "atlas2highres_warp{FSLOUTTYPE}"
#         self.exist = self.out_warp.is_file()

#     def invert_warp(self):
#         invwarp = reg_functions.invert_warp(
#             self.in_file, self.ref, self.out_warp, self.warp
#         )
#         invwarp.run()

#     def run(self):
#         if not self.exist:
#             print("Inverting highres2standard warp...")
#             self.invert_warp()
#         else:
#             print("Inverted highres2standard warp already exists. Continuing.")
#         return self.out_warp


# class ApplyWarp:
#     def __init__(self, warp: Path, in_file: Path, ref: Path, out_dir: Path):
#         self.warp = warp
#         self.in_file = in_file
#         self.ref = ref
#         self.out_dir = out_dir
#         if "highres.nii" in str(in_file):
#             atlas_type = "highres"
#         elif "Labels.nii" in str(in_file):
#             atlas_type = "labels"
#         self.out_file = out_dir / f"{atlas_type}_atlas2highres{FSLOUTTYPE}"
#         self.exist = self.out_file.is_file()

#     def apply_warp(self):
#         aw = reg_functions.apply_warp(self.in_file, self.ref, self.warp, self.out_file)
#         aw.run()

#     def run(self):
#         if not self.exist:
#             print("Apply non-linear warp on atlas file")
#             self.apply_warp()
#         else:
#             print("Warped atlas file already exists. Continuing.")
#         return self.out_file


# class Resample2DWI:
#     def __init__(
#         self, epi: Path, standard_in_highres: Path, out_dir: Path, epi_type: str
#     ):
#         self.epi = epi
#         self.standard_in_highres = standard_in_highres
#         self.out_dir = out_dir
#         self.epi_type = epi_type
#         if "highres_atlas" in str(standard_in_highres):
#             atlas_type = "highres"
#         elif "labels_atlas" in str(standard_in_highres):
#             atlas_type = "labels"
#         self.out_file = out_dir / f"{atlas_type}_atlas2{epi_type}{FSLOUTTYPE}"
#         self.exist = self.out_file.is_file()
#         self.out_matrix_file = out_dir / f"{atlas_type}_atlas2{epi_type}_affine.mat"

#     def resample_2_dwi(self):
#         flt = reg_functions.highres2dwi(
#             self.standard_in_highres, self.epi, self.out_file, self.out_matrix_file
#         )
#         flt.run()

#     def run(self):
#         if not self.exist:
#             print(f"Resampling atlas from highres space to {self.epi_type} image...")
#             self.resample_2_dwi()
#         else:
#             print(f"Atlas in {self.epi_type} space already exists. Continuing.")
#         return self.out_file


# class ApplyAffine:
#     def __init__(self, dwi: Path, labels: Path, aff: Path, out_dir: Path):
#         self.dwi = dwi
#         self.labels = labels
#         self.aff = aff
#         self.out_dir = out_dir
#         self.out_file = out_dir / f"labels_atlas2dwi{FSLOUTTYPE}"
#         self.exist = self.out_file.is_file()

#     def apply_affine(self):
#         ax = reg_functions.apply_affine(self.dwi, self.labels, self.aff, self.out_file)
#         ax.run()

#     def run(self):
#         if not self.exist:
#             print("Applying affine transformation on labels image...")
#             self.apply_affine()
#         else:
#             print("Labels image already exist. Continuing.")


# class NativeSpaceAtlas:
#     def __init__(
#         self,
#         derivatives_dir: Path,
#         atlas: Path = ATLAS,
#         subj: str = None,
#         out_dir: Path = None,
#     ):
#         for f in atlas.iterdir():
#             if "highres.nii" in str(f):
#                 self.highres_atlas = f
#             elif "Labels.nii" in str(f):
#                 self.labels_atlas = f
#         self.derivatives = derivatives_dir
#         self.subjects = []
#         if subj:
#             self.subjects.append(subj)
#         else:
#             self.subjects = [
#                 Path(cur_subj).name
#                 for cur_subj in glob.glob(f"{self.derivatives}/sub-*")
#             ]
#         self.subjects.sort()
#         self.out_dir = []
#         if out_dir:
#             self.out_dir.append(out_dir)
#         else:
#             for subj in self.subjects:
#                 self.out_dir.append(self.derivatives / subj / "atlases")

#     def load_files(self, subj: str):
#         """[summary]

#             Arguments:
#                 subj {str} -- [description]
#             """
#         anat_dir = self.derivatives / subj / "anat" / "prep.anat"
#         standard2highres_warp = anat_dir / f"MNI_to_T1_nonlin_field{FSLOUTTYPE}"
#         highres_brain = anat_dir / f"T1_biascorr_brain{FSLOUTTYPE}"
#         dwi_dir = self.derivatives / subj / "dwi"
#         for f in dwi_dir.iterdir():
#             if "meanbzero" in str(f):
#                 dwi = f
#         feat_dir = Path(self.derivatives / subj / "func" / "func.feat")
#         mean_func = feat_dir / f"mean_func{FSLOUTTYPE}"
#         highres2standard_warp = feat_dir / "reg" / f"highres2standard_warp{FSLOUTTYPE}"

#         return (
#             standard2highres_warp,
#             highres2standard_warp,
#             highres_brain,
#             dwi,
#             mean_func,
#         )

#     def apply_warp(self, warp: Path, in_file: Path, ref: Path, out_dir: Path):
#         aw = ApplyWarp(warp, in_file, ref, out_dir)
#         aw.run()

#     def resample_to_dwi(
#         self, epi: Path, standard_in_highres: Path, out_dir: Path, epi_type: str
#     ):
#         resample = Resample2DWI(epi, standard_in_highres, out_dir, epi_type)
#         resample.run()

#     def run(self):
#         for subj, out_dir in zip(self.subjects, self.out_dir):
#             standard2highres_warp, highres2standard_warp, highres_brain, dwi, mean_func = self.load_files(
#                 subj
#             )
#             atlas2highres, labels2highres = [
#                 self.apply_warp(standard2highres_warp, atlas, highres_brain, out_dir)
#                 for atlas in [self.highres_atlas, self.labels_atlas]
#             ]
#             [
#                 reg_functions.RoundAtlas(atlas)
#                 for atlas in [atlas2highres, labels2highres]
#             ]
#             atlas2dwi, labels2dwi = [
#                 self.resample_to_dwi(dwi, atlas, out_dir, "dwi")
#                 for atlas in [atlas2highres, labels2highres]
#             ]
#             atlas2func, labels2func = [
#                 self.resample_to_dwi(mean_func, atlas, out_dir, "func")
#                 for atlas in [atlas2highres, labels2highres]
#             ]
#             [
#                 reg_functions.RoundAtlas(atlas)
#                 for atlas in [atlas2dwi, labels2dwi, atlas2func, labels2func]
#             ]


class SubjectsAtlas:
    def __init__(
        self,
        derivatives_dir: Path,
        atlas: Path = ATLAS,
        subj: str = None,
        out_dir: Path = None,
        use_feat: bool = True,
    ):
        """

        Arguments:
            derivatives_dir {Path} -- [Path to derivatives directory (preprocessed using PyPrep.code.Preprocessing)]

        Keyword Arguments:
            atlas {Path} -- [Path to atlas' directory (must contain highres and labels files)] (default: {megaatlas})
            subj {str} -- ["sub-xx" for specific subject or None for all subjects] (default: {None (i.e all subjects)})
            out_dir {Path} -- [Output directory] (default: {"{derivatives_dir}/sub-xx/atlases)
        """
        for f in atlas.iterdir():
            if "highres.nii" in str(f):
                self.highres_atlas = f
            elif "Labels.nii" in str(f):
                self.labels_atlas = f
        self.derivatives = derivatives_dir
        self.subjects = []
        if subj:
            self.subjects.append(subj)
        else:
            self.subjects = [
                Path(cur_subj).name
                for cur_subj in glob.glob(f"{self.derivatives}/sub-*")
            ]
        self.subjects.sort()
        self.out_dir = []
        if out_dir:
            self.out_dir.append(out_dir)
        else:
            for subj in self.subjects:
                self.out_dir.append(self.derivatives / subj / "atlases")
        self.use_feat = use_feat

    def load_files(self, subj: str):
        """[summary]

        Arguments:
            subj {str} -- [description]
        """
        anat_dir = self.derivatives / subj / "anat" / "prep.anat"
        standard2highres_warp = anat_dir / f"MNI_to_T1_nonlin_field{FSLOUTTYPE}"
        highres_brain = anat_dir / f"T1_biascorr_brain{FSLOUTTYPE}"
        dwi_dir = self.derivatives / subj / "dwi"
        for f in dwi_dir.iterdir():
            if "meanbzero" in str(f):
                dwi = f
        func = self.derivatives / subj / "func" / "func.feat" / f"mean_func{FSLOUTTYPE}"
        return standard2highres_warp, highres_brain, dwi, func

    def get_transofrm(self, subj: str):
        """
        Initiate the relevant transforms from FEAT preprocessing dir
        Arguments:
            subj {str} -- [sub-xx in the derivatives directory]
        """
        reg_dir = Path(self.derivatives / subj / "func" / "func.feat" / "reg")
        highres2standard_warp = reg_dir / f"highres2standard_warp{FSLOUTTYPE}"
        return highres2standard_warp

    def invert_warp(self, in_file: Path, ref: Path, out_dir: Path, warp: Path):
        """
        Invert the highres2standard warp file
        Arguments:
            warp {Path} -- [highres to standard warp file]
        """
        out_warp = out_dir / f"atlas2highres_warp{FSLOUTTYPE}"
        if not out_warp.is_file():
            print("Inverting highres2standard warp...")
            invwarp = reg_functions.invert_warp(in_file, ref, out_warp, warp)
            print(invwarp.cmdline)
            invwarp.run()
        return out_warp

    def apply_warp(self, warp: Path, in_file: Path, ref: Path, out_dir: Path):
        """[summary]

        Arguments:
            warp {Path} -- [description]
            in_file {Path} -- [description]
            ref {Path} -- [description]
            out_dir {Path} -- [description]
        """
        if "highres.nii" in str(in_file):
            atlas_type = "highres"
        elif "Labels.nii" in str(in_file):
            atlas_type = "labels"
        out_file = out_dir / f"{atlas_type}_atlas2highres{FSLOUTTYPE}"
        if not out_file.is_file():
            print("Apply non-linear warp on atlas file")
            aw = reg_functions.apply_warp(in_file, ref, warp, out_file)
            print(aw.cmdline)
            aw.run()
        return out_file

    def resample_to_dwi(
        self, epi: Path, standard_in_highres: Path, out_dir: Path, epi_type: str
    ):
        """[summary]

        Arguments:
            dwi {Path} -- [description]
            standard_in_highres {Path} -- [description]
            out_dir {Path} -- [description]
        """
        if "highres_atlas" in str(standard_in_highres):
            atlas_type = "highres"
        elif "labels_atlas" in str(standard_in_highres):
            atlas_type = "labels"
        out_file = out_dir / f"{atlas_type}_atlas2{epi_type}{FSLOUTTYPE}"
        out_matrix_file = out_dir / f"{atlas_type}_atlas2{epi_type}_affine.mat"
        if not out_file.is_file():
            print(f"Resampling atlas from highres space to {epi_type} image...")
            flt = reg_functions.highres2dwi(
                standard_in_highres, epi, out_file, out_matrix_file
            )
            print(flt.cmdline)
            flt.run()
        return out_file

    def apply_affine(self, dwi: Path, labels: Path, aff: Path, out_dir: Path):
        """[summary]

        Arguments:
            dwi {Path} -- [description]
            labels {Path} -- [description]
            aff {Path} -- [description]
        """
        out_file = out_dir / f"labels_atlas2dwi{FSLOUTTYPE}"
        if not out_file.is_file():
            print("Applying affine transformation on labels image...")
            ax = reg_functions.apply_affine(dwi, labels, aff, out_file)
            print(ax.cmdline)
            ax.run()

    def run(self):
        for subj, out_dir in zip(self.subjects, self.out_dir):
            warp, highres_brain, dwi, func = self.load_files(subj)
            atlas2highres, labels2highres = [
                self.apply_warp(warp, atlas, highres_brain, out_dir)
                for atlas in [self.highres_atlas, self.labels_atlas]
            ]
            atlas2dwi, labels2dwi = [
                self.resample_to_dwi(dwi, atlas, out_dir, "dwi")
                for atlas in [atlas2highres, labels2highres]
            ]
            reg_functions.RoundAtlas(labels2highres)
            atlas2func, labels2func = [
                self.resample_to_dwi(func, atlas, out_dir, "func")
                for atlas in [atlas2highres, labels2highres]
            ]
            reg_functions.RoundAtlas(labels2highres)
            [
                reg_functions.RoundAtlas(atlas)
                for atlas in [
                    atlas2highres,
                    labels2highres,
                    atlas2dwi,
                    labels2dwi,
                    atlas2func,
                    labels2func,
                ]
            ]
            # self.apply_affine(dwi, labels2highres, aff, out_dir)


if __name__ == "__main__":
    derivatives = Path("/Users/dumbeldore/Desktop/derivatives")
    sa = SubjectsAtlas(derivatives)
    sa.run()

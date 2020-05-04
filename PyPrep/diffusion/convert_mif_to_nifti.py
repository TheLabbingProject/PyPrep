from pathlib import Path
from logs import messages
from PyPrep.diffusion import dmri_prep_functions as dmri_methods
from PyPrep.utils import check_files_existence, FSLOUTTYPE


class ConvertMif2Nifti:
    def __init__(self, derivatives: Path, subj: str):
        self.subj = subj
        self.mrtrix_dir = Path(derivatives / subj / "dwi" / "Mrtrix_prep")
        self.anat = Path(derivatives / subj / "anat")
        self.dwi = self.mrtrix_dir.parent

    def __str__(self):
        str_to_print = messages.CONVERT2NIFTI.format(
            mrtrix_dir=self.mrtrix_dir, subj_dir=self.anat.parent
        )
        return str_to_print

    def convert_files(self):
        for f in self.mrtrix_dir.iterdir():
            if f.is_file():
                f_name = f.name
                if (
                    "fieldmap" in f_name or "b0" in f_name or "PA" in f_name
                ) and "mif" in f_name:
                    #                    f.unlink()
                    continue
                elif (
                    ("T1" in f_name or "vis" in f_name or "5TT" in f_name)
                    and "mif" in f_name
                    and "dwi" not in f_name
                ):
                    out_file = self.anat / f"{f.stem}{FSLOUTTYPE}"
                    if not out_file.is_file():
                        print(f"Converting {f_name} to {out_file}.")
                        dmri_methods.convert_to_mif(f, out_file)
                elif ("dwi" in f_name or "AP" in f_name) and "mif" in f_name:
                    if "preprocessed" in f_name:
                        out_file = self.dwi / f"{self.subj}_acq-AP_{f.stem}{FSLOUTTYPE}"
                    else:
                        out_file = self.dwi / f"{f.stem}{FSLOUTTYPE}"
                    if not out_file.is_file():
                        print(f"Converting {f_name} to {out_file}.")
                        dmri_methods.convert_to_mif(f, out_file)

    def run(self):
        print(
            "Converting Mrtrix3's .mif files to the more commonly used .nii.gz format."
        )
        self.convert_files()

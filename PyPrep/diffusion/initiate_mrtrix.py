from pathlib import Path
from logs import messages
from PyPrep.diffusion import dmri_prep_functions as dmri_methods


class InitiateMrtrix:
    """
    Convert niftis to Mrtrix`s .mif files
    Arguments:
        mrt_folder {Path} -- [Path to subjects' Mrtrix preprocessing directory]
        dwi {Path} -- [Path to motion corrected dwi file]
        mask {Path} -- [Path to field magnitude`s brain mask]
        anat {Path} -- [Path to subject`s strctural image]
        bvec {Path} -- [Path to dwi`s .bvec file]
        bval {Path} -- [Path to dwi`s .bval file]
        phasediff {Path} -- [Path to opposite-phased dwi image]
    """

    def __init__(
        self,
        subj: str,
        derivatives: Path,
        dwi: Path,
        mask: Path,
        anat: Path,
        bvec: Path,
        bval: Path,
        phasediff: Path,
    ):
        self.subj = subj
        self.mrtrix_dir = Path(derivatives / subj / "dwi" / "Mrtrix_prep")
        self.dir_exists = self.mrtrix_dir.is_dir()
        self.dwi = dwi
        self.mask = mask
        self.anat = anat
        self.bvec = bvec
        self.bval = bval
        self.phasediff = phasediff

    def __str__(self):
        str_to_print = messages.INITIATEMRTRIX.format(
            mrtrix_dir=self.mrtrix_dir,
            dwi=self.dwi.name,
            anat=self.anat.name,
            phasediff=self.phasediff.name,
        )
        return str_to_print

    def transfer_files_to_mrt(self):
        files_list = [self.anat, self.dwi, self.mask, self.phasediff]
        print("Converting files to .mif format...")
        for f in files_list:
            f_name = Path(f.stem).stem + ".mif"
            new_f = Path(self.mrtrix_dir / f_name)
            if not new_f.is_file():
                if "T1" in str(f):
                    print("Importing T1 image into temporary directory")
                    new_anat = dmri_methods.convert_to_mif(f, new_f)
                elif "mask" in str(f):
                    print("Importing mask image into temporary directory")
                    new_mask = dmri_methods.convert_to_mif(f, new_f)
                else:
                    if "AP" in str(f):
                        print("Importing DWI data into temporary directory")
                        new_dwi = dmri_methods.convert_to_mif(
                            f, new_f, self.bvec, self.bval
                        )
                    elif "PA" in str(f):
                        print(
                            "Importing reversed phased encode data into temporary directory"
                        )
                        new_PA = dmri_methods.convert_to_mif(f, new_f)
            else:
                if "T1" in str(f):
                    new_anat = new_f
                elif "mask" in str(f):
                    new_mask = new_f
                else:
                    if "AP" in str(f):
                        new_dwi = new_f
                    elif "PA" in str(f):
                        new_PA = new_f
        self.new_anat, self.new_dwi, self.new_mask, self.new_PA = (
            new_anat,
            new_dwi,
            new_mask,
            new_PA,
        )

    def run(self):
        if not self.dir_exists:
            print("Initiate Mrtrix preprocessing directory.")
            self.mrtrix_dir.mkdir()
        else:
            print("Mrtrix preprocessing directory already exists. Continuing.")
        self.transfer_files_to_mrt()
        return self.mrtrix_dir, self.new_anat, self.new_dwi, self.new_mask, self.new_PA
